#!/usr/bin/env python3
"""End-to-end regression harness for pre-first-event stream failover.

An Responses upstream may answer with HTTP 200 and then fail *inside* the SSE
stream (`response.failed`, `response.incomplete` or `error`). Before the
StartGate change in `rust/uni-api-native/src/bin/uni-api-front/provider_stream.rs`
the gateway had already committed that 200 to the client, so the failure reached
the client as a broken stream and same-tier failover could not run.

This harness reproduces both outcomes against a real container image, with no
external provider, no network dependency and no credential:

* ``failover``   - a synthetic upstream that always fails before its first event
  is followed by a synthetic healthy upstream; the client must receive a normal
  completion produced by the second provider, and the log must record one
  structured failure plus a failed 502 attempt followed by a completed attempt.
* ``route-exhausted`` - with the failing upstream alone, the client must receive
  a real HTTP 502 JSON error carrying the upstream reason, never a 200 stream
  with an embedded error object.

Usage::

    python3 test/stream_failover_e2e.py                 # uses deploy/compose.yaml image
    python3 test/stream_failover_e2e.py --image local/newt-uni-api:tag
    python3 test/stream_failover_e2e.py --build         # build rust-runtime first

Requires a working docker daemon and network access to pull ``python:3.11-slim``
only on a cold host.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_IMAGE = "python:3.11-slim"
UPSTREAM_PORT = 9099
GATEWAY_PORT = 4010
GATEWAY_KEY = "e2e-gateway-key"
FAILURE_CODE = "rate_limit_exceeded"
FAILURE_MESSAGE = "synthetic early stream failure"
HEALTHY_TEXT = "healthy-fallback-ok"

UPSTREAM_SOURCE = '''\
"""Synthetic Responses upstreams used by test/stream_failover_e2e.py."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAILURE = (
    b'data: {"type":"response.failed","response":{"id":"resp_failing","status":"failed",'
    b'"error":{"code":"CODE","type":"usage_limit","message":"MESSAGE"}}}\\n\\n'
)
HEALTHY = (
    b'data: {"type":"response.created","response":{"id":"resp_healthy","status":"in_progress"}}\\n\\n'
    b'data: {"type":"response.output_text.delta","delta":"TEXT"}\\n\\n'
    b'data: {"type":"response.completed","response":{"id":"resp_healthy","status":"completed",'
    b'"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed",'
    b'"content":[{"type":"output_text","text":"TEXT"}]}],'
    b'"usage":{"input_tokens":1,"output_tokens":3,"total_tokens":4}}}\\n\\n'
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _drain(self):
        if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                size = int(line.split(b";")[0], 16)
                if size == 0:
                    self.rfile.readline()
                    break
                self.rfile.read(size)
                self.rfile.readline()
        else:
            length = int(self.headers.get("content-length") or 0)
            if length:
                self.rfile.read(length)

    def do_POST(self):
        self._drain()
        body = FAILURE if "/failing/" in self.path else HEALTHY
        print("served", self.path, len(body), flush=True)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = run(["docker", *args])
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def compose_image() -> str:
    for line in (REPO_ROOT.parent / "deploy" / "compose.yaml").read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("image:"):
            return stripped.split(":", 1)[1].strip()
    raise RuntimeError("no image: entry found in deploy/compose.yaml")


def build_image(tag: str) -> None:
    print(f"building {tag} (rust-runtime target)")
    result = run(
        ["docker", "build", "--target", "rust-runtime", "-t", tag,
         "--build-arg", "SOURCE_COMMIT=e2e-stream-failover", str(REPO_ROOT)]
    )
    if result.returncode != 0:
        raise RuntimeError(f"image build failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def ensure_upstream_image() -> None:
    if docker("image", "inspect", UPSTREAM_IMAGE, check=False).returncode != 0:
        print(f"pulling {UPSTREAM_IMAGE}")
        docker("pull", UPSTREAM_IMAGE)


def wait_for_health(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"gateway did not become healthy at {url}")


def chat_completion(port: int, prompt: str = "Reply with exactly: OK") -> tuple[int, str]:
    payload = json.dumps({
        "model": "tier-2",
        "stream": True,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def provider(name: str, path: str, model: str, upstream_host: str) -> str:
    return (
        f"- provider: {name}\n"
        f"  engine: gpt\n"
        f"  base_url: http://{upstream_host}:{UPSTREAM_PORT}{path}/v1/responses\n"
        f"  api:\n"
        f"  - synthetic-key\n"
        f"  model:\n"
        f"  - {model}: tier-2\n"
    )


def write_config(directory: Path, providers: str) -> Path:
    path = directory / "api.yaml"
    path.write_text(
        "providers:\n"
        f"{providers}"
        "api_keys:\n"
        f'- api: "{GATEWAY_KEY}"\n'
        "  model:\n"
        "  - tier-2\n"
    )
    return path


def start_gateway(name: str, network: str, image: str, config: Path, workdir: Path) -> None:
    data_dir = workdir / "data"
    data_dir.mkdir(exist_ok=True)
    docker(
        "run", "-d", "--name", name, "--network", network,
        "-p", f"127.0.0.1:{GATEWAY_PORT}:8000",
        "-v", f"{config}:/home/api.yaml:ro",
        "-v", f"{data_dir}:/home/data",
        image,
    )


def gateway_log(name: str) -> str:
    return docker("logs", name).stdout + docker("logs", name).stderr


def scenario_failover(image: str, network: str, workdir: Path, upstream_host: str) -> list[str]:
    failures: list[str] = []
    providers = (
        provider("e2e-failing-upstream", "/failing", "e2e-failing-model", upstream_host)
        + provider("e2e-healthy-fallback", "/healthy", "e2e-healthy-model", upstream_host)
    )
    config = write_config(workdir, providers)
    name = "e2e-failover-gateway"
    start_gateway(name, network, image, config, workdir)
    wait_for_health(f"http://127.0.0.1:{GATEWAY_PORT}/healthz")
    try:
        status, body = chat_completion(GATEWAY_PORT)
        logs = gateway_log(name)

        if status != 200:
            failures.append(f"failover: expected HTTP 200, got {status}: {body[:200]}")
        if '"error"' in body:
            failures.append(f"failover: client received an embedded stream error: {body[:200]}")
        if HEALTHY_TEXT not in body:
            failures.append(f"failover: fallback output missing from client stream: {body[:200]}")
        if "[DONE]" not in body:
            failures.append("failover: client stream was not terminated with [DONE]")

        attempts = [
            json.loads(line)
            for line in logs.splitlines()
            if line.startswith("{") and '"rust_native_api_attempt"' in line
        ]
        failed = [a for a in attempts if a.get("outcome") == "failed"]
        completed = [a for a in attempts if a.get("outcome") == "completed"]
        if not failed or failed[0].get("status_code") != 502:
            failures.append(f"failover: expected a 502 attempt failure, got {failed[:1]}")
        if failed and failed[0].get("provider") != "e2e-failing-upstream":
            failures.append(f"failover: unexpected failing provider {failed[:1]}")
        if failed and FAILURE_CODE not in (failed[0].get("detail") or ""):
            failures.append(f"failover: 502 attempt log carries no upstream reason: {failed[:1]}")
        if not any(a.get("provider") == "e2e-healthy-fallback" for a in completed):
            failures.append("failover: fallback provider never completed an attempt")

        failure_events = [
            json.loads(line)
            for line in logs.splitlines()
            if line.startswith("{") and '"rust_upstream_stream_failure"' in line
        ]
        if len(failure_events) != 1:
            failures.append(f"failover: expected exactly one structured failure event, got {len(failure_events)}")
        elif FAILURE_CODE not in failure_events[0].get("reason", ""):
            failures.append(f"failover: failure event lacks the upstream code: {failure_events[0]}")
    finally:
        docker("rm", "-f", name, check=False)
    return failures


def scenario_route_exhausted(image: str, network: str, workdir: Path, upstream_host: str) -> list[str]:
    failures: list[str] = []
    providers = provider("e2e-failing-upstream", "/failing", "e2e-failing-model", upstream_host)
    config = write_config(workdir, providers)
    name = "e2e-exhausted-gateway"
    start_gateway(name, network, image, config, workdir)
    wait_for_health(f"http://127.0.0.1:{GATEWAY_PORT}/healthz")
    try:
        status, body = chat_completion(GATEWAY_PORT)
        if status != 502:
            failures.append(f"route-exhausted: expected HTTP 502, got {status}: {body[:200]}")
        if '"error"' not in body:
            failures.append(f"route-exhausted: response is not a JSON error: {body[:200]}")
        if FAILURE_CODE not in body:
            failures.append(f"route-exhausted: response hides the upstream reason: {body[:200]}")
        if not body.lstrip().startswith("{"):
            failures.append("route-exhausted: an SSE stream leaked to a failed non-streamed response")
    finally:
        docker("rm", "-f", name, check=False)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="gateway image under test (default: deploy/compose.yaml)")
    parser.add_argument("--build", action="store_true", help="build the rust-runtime image from this working tree first")
    args = parser.parse_args()

    if docker("info", check=False).returncode != 0:
        print("docker daemon is not reachable", file=sys.stderr)
        return 2

    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    network = f"e2e-stream-failover-{suffix}"
    upstream = f"e2e-upstream-{suffix}"
    image = f"local/newt-uni-api:e2e-stream-failover-{suffix}" if args.build else (args.image or compose_image())

    if args.build:
        build_image(image)
    elif docker("image", "inspect", image, check=False).returncode != 0:
        print(f"image {image} is not present locally; pass --build to build it", file=sys.stderr)
        return 2

    print(f"image   : {image}")
    print(f"network : {network}")
    failures: list[str] = []
    try:
        ensure_upstream_image()
        docker("network", "create", network)
        with tempfile.TemporaryDirectory(prefix="uni-api-e2e-") as raw_workdir:
            workdir = Path(raw_workdir)
            source = workdir / "upstream.py"
            source.write_text(
                UPSTREAM_SOURCE
                .replace("PORT", str(UPSTREAM_PORT))
                .replace("CODE", FAILURE_CODE)
                .replace("MESSAGE", FAILURE_MESSAGE)
                .replace("TEXT", HEALTHY_TEXT)
            )
            docker(
                "run", "-d", "--name", upstream, "--network", network,
                "-v", f"{source}:/app/upstream.py:ro",
                UPSTREAM_IMAGE, "python", "/app/upstream.py",
            )
            time.sleep(2)
            status = docker("inspect", "-f", "{{.State.Status}}", upstream, check=False).stdout.strip()
            if status != "running":
                raise RuntimeError(
                    f"synthetic upstream is {status or 'absent'}: "
                    + (docker("logs", upstream, check=False).stdout
                       + docker("logs", upstream, check=False).stderr)[-500:]
                )
            failures += scenario_failover(image, network, workdir, upstream)
            failures += scenario_route_exhausted(image, network, workdir, upstream)
    except Exception as error:  # noqa: BLE001 - harness reports any failure as a test failure
        failures.append(f"harness error: {error}")
    finally:
        docker("rm", "-f", upstream, check=False)
        docker("network", "rm", network, check=False)

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS  failover: early upstream stream failure is retried on the next same-tier provider")
    print("PASS  route-exhausted: client receives HTTP 502 with the real upstream reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
