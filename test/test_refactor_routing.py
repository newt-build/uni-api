import asyncio
import contextlib
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from fastapi import BackgroundTasks
from fastapi import HTTPException
from starlette.responses import Response, StreamingResponse
from core.models import RequestModel
from routing import RoutingPlan, build_api_key_models_map, get_right_order_providers
from uni_api.admission import (
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.upstream.responses_errors import responses_failure_error


def test_build_api_key_models_map_resolves_nested_api_keys():
    config = {
        "providers": [
            {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1/chat/completions",
                "model": ["gpt-4.1", "gpt-4o-mini"],
            },
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com/v1/messages",
                "model": ["claude-sonnet-4-5"],
            },
        ],
        "api_keys": [
            {
                "api": "sk-root",
                "model": ["openai/*"],
            },
            {
                "api": "sk-nested",
                "model": ["sk-root/*", "anthropic/claude-sonnet-4-5"],
            },
        ],
    }

    models_map = build_api_key_models_map(config, ["sk-root", "sk-nested"])

    assert models_map["sk-root"] == ["gpt-4.1", "gpt-4o-mini"]
    assert models_map["sk-nested"] == ["gpt-4.1", "gpt-4o-mini", "claude-sonnet-4-5"]


def test_get_right_order_providers_filters_provider_excluded_endpoint():
    config = {
        "providers": [
            {
                "provider": "provider-a",
                "base_url": "https://provider-a.example/v1/responses",
                "model": ["gpt-5.4"],
                "exclude_endpoints": ["v1/responses/compact"],
            },
            {
                "provider": "provider-b",
                "base_url": "https://provider-b.example/v1/responses",
                "model": ["gpt-5.4"],
            },
            {
                "provider": "provider-c",
                "base_url": "https://provider-c.example/v1/responses",
                "model": ["gpt-5.4"],
                "preferences": {
                    "exclude_endpoints": ["/v1/responses/compact/"],
                },
            },
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
            }
        ],
    }

    compact_providers = asyncio.run(
        get_right_order_providers(
            "gpt-5.4",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.4"]},
            endpoint="/v1/responses/compact",
        )
    )
    regular_providers = asyncio.run(
        get_right_order_providers(
            "gpt-5.4",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.4"]},
            endpoint="/v1/responses",
        )
    )

    assert [provider["provider"] for provider in compact_providers] == ["provider-b"]
    assert [provider["provider"] for provider in regular_providers] == [
        "provider-a",
        "provider-b",
        "provider-c",
    ]


def test_routing_plan_passes_endpoint_to_provider_resolver():
    received = {}

    async def fake_resolver(
        request_model_name,
        config,
        api_index,
        scheduling_algorithm,
        api_list,
        models_list,
        *,
        endpoint=None,
        **kwargs,
    ):
        _ = config, api_index, scheduling_algorithm, api_list, models_list, kwargs
        received["endpoint"] = endpoint
        return [
            {
                "provider": "provider-a",
                "_model_dict_cache": {request_model_name: request_model_name},
                "base_url": "https://provider-a.example/v1/responses",
                "api": None,
                "preferences": {},
            }
        ]

    config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
            }
        ]
    }
    app = SimpleNamespace(
        state=SimpleNamespace(
            config=config,
            api_list=["sk-test"],
            models_list={"sk-test": ["gpt-5.4"]},
            channel_manager=None,
        )
    )

    asyncio.run(
        RoutingPlan.create(
            app,
            "gpt-5.4",
            0,
            {},
            {},
            endpoint="/v1/responses/compact",
            provider_resolver=fake_resolver,
        )
    )

    assert received["endpoint"] == "/v1/responses/compact"


def test_client_manager_reuses_single_client_under_concurrency(monkeypatch):
    created_clients = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_clients.append(self)

        async def aclose(self):
            return None

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)

    async def run_test():
        manager = main.ClientManager(pool_size=4)
        await manager.init(
            {
                "headers": {"User-Agent": "test"},
                "http2": True,
                "verify": True,
                "follow_redirects": True,
            }
        )

        async def borrow_client():
            async with manager.get_client("https://example.com/v1/chat/completions") as client:
                await asyncio.sleep(0)
                return client

        clients = await asyncio.gather(*[borrow_client() for _ in range(20)])
        assert len(created_clients) == 1
        assert len({id(client) for client in clients}) == 1
        await manager.close()

    asyncio.run(run_test())


def test_client_manager_separates_proxy_and_http2_keys(monkeypatch):
    created_clients = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created_clients.append(self)

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)

    async def run_test():
        manager = main.ClientManager(pool_size=4)
        await manager.init(
            {
                "headers": {"User-Agent": "test"},
                "http2": True,
                "verify": True,
                "follow_redirects": True,
            }
        )

        async with manager.get_client("https://example.com/v1/chat/completions", proxy=None, http2=True) as client_a:
            pass
        async with manager.get_client("https://example.com/v1/chat/completions", proxy="socks5h://127.0.0.1:1080", http2=True) as client_b:
            pass
        async with manager.get_client("https://example.com/v1/chat/completions", proxy=None, http2=False) as client_c:
            pass

        assert len({id(client_a), id(client_b), id(client_c)}) == 3
        assert manager.snapshot()["client_count"] == 3
        await manager.close()
        assert manager.clients == {}
        assert all(client.closed for client in created_clients)

    asyncio.run(run_test())


def test_process_request_uses_http1_client_for_codex_chat(monkeypatch):
    class DummyClient:
        pass

    class DummyClientManager:
        def __init__(self):
            self.calls = []

        @contextlib.asynccontextmanager
        async def get_client(self, base_url, proxy=None, http2=None):
            self.calls.append({"base_url": base_url, "proxy": proxy, "http2": http2})
            yield DummyClient()

    async def fake_fetch_response_stream(client, url, headers, payload, engine, model, timeout, response_headers_sink=None):
        _ = (client, url, headers, payload, engine, model, timeout, response_headers_sink)
        yield (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,'
            '"model":"gpt-image-2","choices":[{"index":0,"delta":{"role":"assistant"},'
            '"finish_reason":null}]}\n\n'
        )

    client_manager = DummyClientManager()
    monkeypatch.setattr(main, "fetch_response_stream", fake_fetch_response_stream)
    main.app.state.client_manager = client_manager
    main.app.state.config = {"preferences": {}}
    main.app.state.error_triggers = []

    async def run_test():
        token = main.request_info.set(
            {
                "request_id": "req-test",
                "api_key": "sk-test",
                "first_response_time": None,
                "success": False,
                "provider": None,
            }
        )
        try:
            response = await main.process_request(
                RequestModel(
                    model="gpt-image-2",
                    messages=[{"role": "user", "content": "draw"}],
                    stream=True,
                ),
                {
                    "provider": "fugue-codex",
                    "engine": "codex",
                    "base_url": "https://oaix.fugue.pro/v1/responses",
                    "api": "change-me",
                    "model": ["gpt-image-2"],
                    "_model_dict_cache": {"gpt-image-2": "gpt-image-2"},
                    "preferences": {},
                    "tools": True,
                },
                BackgroundTasks(),
                role="admin",
                timeout_value=30,
                provider_api_key_raw="change-me",
            )
            assert response.status_code == 200
            assert client_manager.calls == [
                {
                    "base_url": "https://oaix.fugue.pro/v1/responses",
                    "proxy": None,
                    "http2": False,
                }
            ]
        finally:
            main.request_info.reset(token)

    asyncio.run(run_test())


def test_process_request_collects_nonstream_codex_chat_completion(monkeypatch):
    class DummyClient:
        pass

    class DummyClientManager:
        @contextlib.asynccontextmanager
        async def get_client(self, base_url, proxy=None, http2=None):
            _ = (base_url, proxy, http2)
            yield DummyClient()

    async def fake_fetch_response(*_args, **_kwargs):
        raise AssertionError("non-stream transport must not be used for Codex chat")

    async def fake_fetch_response_stream(
        _client, _url, _headers, payload, _engine, _model, _timeout,
        response_headers_sink=None,
    ):
        assert payload["stream"] is True
        assert payload["max_output_tokens"] == 42
        if response_headers_sink is not None:
            response_headers_sink({"x-upstream": "test"})
        yield (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"gpt-5.6-luna","choices":[{'
            '"index":0,"delta":{"role":"assistant","content":"ok"},'
            '"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"gpt-5.6-luna","choices":[{'
            '"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        yield "data: [DONE]\n\n"

    client_manager = DummyClientManager()
    monkeypatch.setattr(main, "fetch_response", fake_fetch_response)
    monkeypatch.setattr(main, "fetch_response_stream", fake_fetch_response_stream)
    main.app.state.client_manager = client_manager
    main.app.state.config = {"preferences": {}}
    main.app.state.error_triggers = []

    async def run_test():
        token = main.request_info.set(
            {
                "request_id": "req-codex-chat-nonstream",
                "api_key": "sk-test",
                "first_response_time": None,
                "success": False,
                "provider": None,
            }
        )
        try:
            response = await main.process_request(
                RequestModel(
                    model="gpt-5.6-luna",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=False,
                    max_tokens=42,
                ),
                {
                    "provider": "codex-a",
                    "engine": "codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "api": "change-me",
                    "model": ["gpt-5.6-luna"],
                    "_model_dict_cache": {"gpt-5.6-luna": "gpt-5.6-luna"},
                    "preferences": {},
                    "tools": True,
                },
                BackgroundTasks(),
                endpoint="/v1/chat/completions",
                role="admin",
                timeout_value=30,
                provider_api_key_raw="change-me",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
            body = b"".join(chunks)
            payload = json.loads(body)
            assert response.media_type == "application/json"
            assert payload["object"] == "chat.completion"
            assert payload["choices"][0]["message"]["content"] == "ok"
        finally:
            main.request_info.reset(token)

    asyncio.run(run_test())

def test_model_request_handler_passes_selected_provider_key(monkeypatch):
    provider_name = "provider-a"

    class DummyResponseMemoryLease:
        def begin_response_attempt(self, *_args, **_kwargs):
            return None

        def finish_response_attempt(self, *_args, **_kwargs):
            return None

    response_memory_lease = DummyResponseMemoryLease()

    class DummyCircularList:
        async def is_all_rate_limited(self, model):
            return False

        async def next(self, model):
            return "provider-key-1"

        def get_items_count(self):
            return 1

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-4.1": "gpt-4.1"},
                "base_url": "https://example.com/v1/chat/completions",
                "api": ["provider-key-1"],
                "preferences": {},
            }
        ]

    async def fake_process_request(
        request,
        provider,
        background_tasks,
        endpoint=None,
        role=None,
        timeout_value=0,
        keepalive_interval=None,
        provider_api_key_raw=None,
        current_info=None,
        http_request=None,
        response_memory_lease=None,
    ):
        _ = current_info, http_request
        assert provider_api_key_raw == "provider-key-1"
        assert response_memory_lease is response_memory_lease_expected
        return Response(content=b"ok", media_type="application/json")

    response_memory_lease_expected = response_memory_lease

    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList())
    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "process_request", fake_process_request)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-4.1"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.keepalive_interval = {"global": {"default": 99999}}

    async def run_test():
        handler = main.ModelRequestHandler()
        token = bind_request_admission_lease(response_memory_lease)
        try:
            response = await handler.request_model(
                RequestModel(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=False,
                ),
                0,
                BackgroundTasks(),
            )
            assert response.status_code == 200
        finally:
            reset_request_admission_lease(token)

    asyncio.run(run_test())


def test_model_request_semantic_context_error_returns_400_without_retry_or_cooldown(
    monkeypatch,
):
    provider_names = ["provider-a", "provider-b"]
    process_calls = []

    class DummyCircularList:
        def __init__(self, key):
            self.key = key
            self.cooling_calls = []

        async def is_all_rate_limited(self, model):
            return False

        async def next(self, model):
            return self.key

        def get_items_count(self):
            return 1

        async def set_cooling(self, item, cooling_time):
            self.cooling_calls.append((item, cooling_time))

    class ChannelManager:
        cooldown_period = 300

        def __init__(self):
            self.excluded = []

        async def exclude_model(self, provider, model):
            self.excluded.append((provider, model))

    lists = {
        provider_name: DummyCircularList(f"{provider_name}-key")
        for provider_name in provider_names
    }
    for provider_name, circular_list in lists.items():
        monkeypatch.setitem(
            main.provider_api_circular_list,
            provider_name,
            circular_list,
        )

    async def fake_get_right_order_providers(
        request_model_name,
        config,
        api_index,
        scheduling_algorithm,
    ):
        _ = config, api_index, scheduling_algorithm
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {request_model_name: request_model_name},
                "base_url": f"https://{provider_name}.example/v1/responses",
                "api": [f"{provider_name}-key"],
                "preferences": {"api_key_cooldown_period": 300},
            }
            for provider_name in provider_names
        ]

    async def fake_process_request(
        request,
        provider,
        background_tasks,
        endpoint=None,
        role=None,
        timeout_value=0,
        keepalive_interval=None,
        provider_api_key_raw=None,
        current_info=None,
        http_request=None,
        response_memory_lease=None,
    ):
        _ = (
            request,
            background_tasks,
            endpoint,
            role,
            timeout_value,
            keepalive_interval,
            provider_api_key_raw,
            current_info,
            http_request,
            response_memory_lease,
        )
        process_calls.append(provider["provider"])
        error = responses_failure_error(
            {
                "error": {
                    "code": "oaix_gateway_error",
                    "message": "Your input exceeds the context window of this model.",
                    "status": 400,
                    "type": "gateway_error",
                }
            },
            event_type="error",
        )
        assert error is not None
        raise error

    channel_manager = ChannelManager()
    monkeypatch.setattr(
        main,
        "get_right_order_providers",
        fake_get_right_order_providers,
    )
    monkeypatch.setattr(main, "process_request", fake_process_request)
    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.5"],
                "preferences": {"AUTO_RETRY": True},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.keepalive_interval = {"global": {"default": 99999}}
    main.app.state.channel_manager = channel_manager

    current_info = {
        "request_id": "semantic-context-error",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    async def run_test():
        handler = main.ModelRequestHandler()
        return await handler.request_model(
            RequestModel(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ),
            0,
            BackgroundTasks(),
            current_info=current_info,
        )

    response = asyncio.run(run_test())

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "error": {
            "message": "Your input exceeds the context window of this model.",
            "status_code": 400,
            "type": "gateway_error",
            "code": "oaix_gateway_error",
        }
    }
    assert process_calls == ["provider-a"]
    assert current_info.get("retry_count", 0) == 0
    assert current_info["attempt_count"] == 1
    assert current_info["retry_decision_count"] == 0
    assert current_info["retry_transition_count"] == 0
    assert current_info["routing_attempts"][0]["semantic_status_code"] == 400
    assert current_info["routing_attempts"][0]["retry_decision"] is False
    assert current_info["routing_attempts"][0]["error_code"] == "oaix_gateway_error"
    assert "error_message" not in current_info["routing_attempts"][0]
    assert channel_manager.excluded == []
    assert all(not circular_list.cooling_calls for circular_list in lists.values())


def test_model_request_same_turn_disconnect_closes_completed_stream_result(
    monkeypatch,
):
    provider_name = "provider-a"
    disconnect_event = asyncio.Event()
    body_closed = False

    class DummyCircularList:
        async def is_all_rate_limited(self, model):
            return False

        async def next(self, model):
            return "provider-key-1"

        def get_items_count(self):
            return 1

    async def fake_get_right_order_providers(*_args, **_kwargs):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-4.1": "gpt-4.1"},
                "base_url": "https://example.com/v1/chat/completions",
                "api": ["provider-key-1"],
                "preferences": {},
            }
        ]

    class Body:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Event().wait()

        async def aclose(self):
            nonlocal body_closed
            body_closed = True

    async def fake_process_request(*_args, **_kwargs):
        # Make process_task and disconnect_task ready in the same event-loop
        # turn.  Disconnect must own the result-transfer race.
        disconnect_event.set()
        return StreamingResponse(Body(), media_type="text/event-stream")

    monkeypatch.setitem(
        main.provider_api_circular_list,
        provider_name,
        DummyCircularList(),
    )
    monkeypatch.setattr(
        main,
        "get_right_order_providers",
        fake_get_right_order_providers,
    )
    monkeypatch.setattr(main, "process_request", fake_process_request)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-4.1"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.keepalive_interval = {"global": {"default": 99999}}

    async def run_test():
        handler = main.ModelRequestHandler()
        response = await handler.request_model(
            RequestModel(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ),
            0,
            BackgroundTasks(),
            current_info={
                "request_id": "same-turn-disconnect",
                "api_key": "sk-test",
                "disconnect_event": disconnect_event,
            },
        )
        assert response.status_code == 499
        assert body_closed is True

    asyncio.run(run_test())


def test_model_request_handler_error_log_includes_request_and_actual_model(monkeypatch):
    provider_name = "provider-a"

    class DummyCircularList:
        async def is_all_rate_limited(self, model):
            return False

        async def next(self, model):
            return "provider-key-1"

        def get_items_count(self):
            return 1

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"friendly-model": "gpt-4.1"},
                "base_url": "https://example.com/v1/chat/completions",
                "api": ["provider-key-1"],
                "preferences": {},
            }
        ]

    async def fake_process_request(
        request,
        provider,
        background_tasks,
        endpoint=None,
        role=None,
        timeout_value=0,
        keepalive_interval=None,
        provider_api_key_raw=None,
        current_info=None,
        http_request=None,
        response_memory_lease=None,
    ):
        _ = current_info, http_request, response_memory_lease
        raise HTTPException(status_code=502, detail="bad gateway")

    error_logs = []

    def fake_error(msg, *args, **kwargs):
        _ = kwargs
        error_logs.append(msg % args if args else msg)

    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList())
    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "process_request", fake_process_request)
    monkeypatch.setattr(main.logger, "error", fake_error)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["friendly-model"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.keepalive_interval = {"global": {"default": 99999}}

    async def run_test():
        handler = main.ModelRequestHandler()
        response = await handler.request_model(
            RequestModel(
                model="friendly-model",
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
            ),
            0,
            BackgroundTasks(),
        )
        assert response.status_code == 502

    asyncio.run(run_test())

    assert any("request_model=friendly-model" in log for log in error_logs)
    assert any("actual_model=gpt-4.1" in log for log in error_logs)
