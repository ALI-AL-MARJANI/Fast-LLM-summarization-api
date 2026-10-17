"""API tests against a mocked vLLM backend (no GPU, no network).

The mock is injected at the transport layer (`httpx.MockTransport`), so the
real `VLLMClient` code (payload, parsing, error mapping) is exercised; only
the network hop to vLLM is replaced.
"""

import json
from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import MAX_TEXT_CHARS

API_KEY = "test-api-key-0123456789"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TEXT = "The quick brown fox jumps over the lazy dog. " * 10

Handler = Callable[[httpx.Request], httpx.Response]


def vllm_ok(request: httpx.Request) -> httpx.Response:
    """Happy-path mock of vLLM's `/health` and `/v1/chat/completions`."""
    if request.url.path == "/health":
        return httpx.Response(200)
    return httpx.Response(
        200,
        json={
            "model": MODEL,
            "choices": [{"message": {"role": "assistant", "content": " A fox jumps. "}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125},
        },
    )


def vllm_down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("Connection refused", request=request)


def vllm_slow(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("Read timed out", request=request)


@pytest.fixture
def make_client() -> Iterator[Callable[[Handler], TestClient]]:
    clients: list[TestClient] = []

    def _make(handler: Handler) -> TestClient:
        settings = Settings(_env_file=None, api_key=API_KEY, vllm_base_url="http://vllm:8000")
        app = create_app(settings, transport=httpx.MockTransport(handler))
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()  # runs the lifespan (creates the shared httpx client)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


def summarize(client: TestClient, payload: dict, api_key: str | None = API_KEY):
    headers = {"X-API-Key": api_key} if api_key is not None else {}
    return client.post("/v1/summarize", json=payload, headers=headers)


def test_summarize_happy_path(make_client):
    response = summarize(make_client(vllm_ok), {"text": TEXT, "style": "bullet_points"})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "A fox jumps."
    assert body["model"] == MODEL
    assert body["usage"] == {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125}
    assert body["latency_ms"] >= 0
    assert response.headers["X-Request-ID"]


def test_summarize_sends_expected_payload(make_client):
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return vllm_ok(request)

    summarize(make_client(handler), {"text": TEXT, "max_tokens": 64})
    sent = captured[0]
    assert sent.url.path == "/v1/chat/completions"
    payload = json.loads(sent.content)
    assert payload["model"] == MODEL
    assert payload["max_tokens"] == 64
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert TEXT.strip() in payload["messages"][1]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": TEXT, "style": "poem"},  # style outside the allowed enum
        {"text": "x" * (MAX_TEXT_CHARS + 1)},  # text too long
        {"text": "   too short   "},  # too short once whitespace is stripped
        {"text": TEXT, "max_tokens": 4096},  # max_tokens above the bound
        {"text": TEXT, "temperature": 1.0},  # unknown field
    ],
)
def test_summarize_invalid_input_returns_422(make_client, payload):
    assert summarize(make_client(vllm_ok), payload).status_code == 422


@pytest.mark.parametrize("api_key", [None, "wrong-key"])
def test_summarize_rejects_missing_or_wrong_api_key(make_client, api_key):
    response = summarize(make_client(vllm_ok), {"text": TEXT}, api_key=api_key)
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing or invalid API key."}


def test_summarize_vllm_down_returns_503(make_client):
    response = summarize(make_client(vllm_down), {"text": TEXT})
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]


def test_summarize_vllm_timeout_returns_504(make_client):
    response = summarize(make_client(vllm_slow), {"text": TEXT})
    assert response.status_code == 504
    assert "in time" in response.json()["detail"]


def test_summarize_vllm_malformed_payload_returns_502(make_client):
    response = summarize(make_client(lambda r: httpx.Response(200, json={})), {"text": TEXT})
    assert response.status_code == 502


def test_health_is_public_and_ready(make_client):
    response = make_client(vllm_ok).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vllm": "ready"}


@pytest.mark.parametrize(
    "handler", [vllm_down, lambda r: httpx.Response(503)], ids=["down", "loading"]
)
def test_health_returns_503_when_vllm_not_ready(make_client, handler):
    response = make_client(handler).get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "vllm": "not_ready"}


def test_unexpected_error_does_not_leak_internals(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("secret internal detail")

    response = summarize(make_client(handler), {"text": TEXT})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}


# --- Streaming (Server-Sent Events) -------------------------------------------

def sse_chunk(data: dict | str) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {payload}\n\n".encode()


USAGE = {"prompt_tokens": 120, "completion_tokens": 5}
STREAM_BODY = b"".join([
    sse_chunk({"model": MODEL, "choices": [{"delta": {"role": "assistant"}}], "usage": None}),
    sse_chunk({"model": MODEL, "choices": [{"delta": {"content": "A fox"}}], "usage": None}),
    sse_chunk({"model": MODEL, "choices": [{"delta": {"content": " jumps."}}], "usage": None}),
    sse_chunk({"model": MODEL, "choices": [], "usage": USAGE}),
    sse_chunk("[DONE]"),
])


class FailingStream(httpx.AsyncByteStream):
    """Sends one token, then times out: simulates vLLM stalling mid-generation."""

    async def __aiter__(self):
        yield sse_chunk({"model": MODEL, "choices": [{"delta": {"content": "A fox"}}]})
        raise httpx.ReadTimeout("Read timed out")


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_happy_path(make_client):
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, content=STREAM_BODY, headers={"Content-Type": "text/event-stream"}
        )

    response = summarize(make_client(handler), {"text": TEXT, "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert captured[0]["stream"] is True
    assert captured[0]["stream_options"] == {"include_usage": True}

    events = parse_sse(response.text)
    assert events[:2] == [("delta", {"text": "A fox"}), ("delta", {"text": " jumps."})]
    name, done = events[2]
    assert name == "done"
    assert done["model"] == MODEL
    assert done["usage"] == {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125}


def test_stream_vllm_down_returns_503_before_streaming(make_client):
    response = summarize(make_client(vllm_down), {"text": TEXT, "stream": True})
    assert response.status_code == 503


def test_stream_upstream_http_error_is_mapped(make_client):
    client = make_client(lambda r: httpx.Response(400))
    response = summarize(client, {"text": TEXT, "stream": True})
    assert response.status_code == 502


def test_stream_mid_generation_timeout_emits_error_event(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=FailingStream())

    response = summarize(make_client(handler), {"text": TEXT, "stream": True})
    assert response.status_code == 200  # headers were already sent
    assert parse_sse(response.text) == [
        ("delta", {"text": "A fox"}),
        ("error", {"detail": "vLLM did not respond in time."}),
    ]


def test_stream_without_usage_emits_error_event(make_client):
    body = b"".join([
        sse_chunk({"model": MODEL, "choices": [{"delta": {"content": "Hi"}}]}),
        sse_chunk("[DONE]"),
    ])
    client = make_client(lambda r: httpx.Response(200, content=body))
    response = summarize(client, {"text": TEXT, "stream": True})
    assert parse_sse(response.text)[-1][0] == "error"
