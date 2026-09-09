from types import SimpleNamespace

from uni_api.providers.execution import ensure_opencode_session_header


def test_preserves_explicit_opencode_session():
    headers = {"x-opencode-session": "hermes-session-1"}
    ensure_opencode_session_header(headers)
    assert headers["x-opencode-session"] == "hermes-session-1"


def test_uses_request_id_when_present():
    headers = {"x-request-id": "request-1"}
    ensure_opencode_session_header(headers)
    assert headers["x-opencode-session"] == "request-1"


def test_uses_incoming_http_session_header():
    headers = {}
    request = SimpleNamespace(headers={"x-session-id": "session-2"})
    ensure_opencode_session_header(headers, http_request=request)
    assert headers["x-opencode-session"] == "session-2"


def test_generates_session_when_missing(monkeypatch):
    monkeypatch.setattr("uni_api.providers.execution.uuid.uuid4", lambda: SimpleNamespace(hex="generated"))
    headers = {}
    ensure_opencode_session_header(headers)
    assert headers["x-opencode-session"] == "generated"
