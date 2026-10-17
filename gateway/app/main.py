"""FastAPI gateway: authentication, validation, error mapping and routes.

Run with: uvicorn app.main:create_app --factory
"""

import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.llm_client import (
    Completion,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    VLLMClient,
)
from app.prompts import build_messages
from app.schemas import HealthResponse, SummarizeRequest, SummarizeResponse, Usage

logger = logging.getLogger("gateway")

ERROR_STATUS: dict[type[LLMError], int] = {
    LLMUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    LLMTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
    LLMResponseError: status.HTTP_502_BAD_GATEWAY,
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_llm(request: Request) -> VLLMClient:
    return request.app.state.llm


def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> None:
    expected = request.app.state.settings.api_key.get_secret_value()
    # Constant-time comparison: response time must not leak how many bytes match.
    if api_key is None or not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def usage_of(completion: Completion) -> Usage:
    return Usage(
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        total_tokens=completion.prompt_tokens + completion.completion_tokens,
    )


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event (`event:` line, `data:` line, blank line)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_summary(
    llm: VLLMClient, upstream: httpx.Response, request_id: str, start: float
) -> AsyncIterator[str]:
    """Re-emit vLLM's token stream as gateway events: `delta`*, then `done` or `error`."""
    try:
        async for item in llm.iter_stream(upstream):
            if isinstance(item, str):
                yield sse("delta", {"text": item})
                continue
            latency_ms = elapsed_ms(start)
            # The access-log line is written when headers are sent, before the
            # body; token counts are only known here, at the end of the stream.
            logger.info(json.dumps({
                "request_id": request_id,
                "event": "stream_complete",
                "latency_ms": latency_ms,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
            }))
            yield sse("done", {
                "model": item.model,
                "usage": usage_of(item).model_dump(),
                "latency_ms": latency_ms,
            })
    except LLMError as exc:
        # The 200 status line is already sent: report the failure in-band.
        logger.warning(json.dumps({"request_id": request_id, "error": str(exc)}))
        yield sse("error", {"detail": str(exc)})


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Application factory. `transport` lets tests replace the network with a mock."""
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One pooled client for the whole process: connections to vLLM are reused
        # (keep-alive) instead of paying a TCP handshake on every request.
        timeout = httpx.Timeout(
            settings.vllm_request_timeout_s, connect=settings.vllm_connect_timeout_s
        )
        async with httpx.AsyncClient(
            base_url=settings.vllm_base_url, timeout=timeout, transport=transport
        ) as http:
            app.state.llm = VLLMClient(http, settings.model_name, settings.temperature)
            yield

    app = FastAPI(title="Fast LLM Summarization API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.log_extra = {}
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            record = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "latency_ms": elapsed_ms(start),
                **request.state.log_extra,
            }
            logger.info(json.dumps(record))

    @app.exception_handler(LLMError)
    async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.warning(json.dumps({"request_id": request_id, "error": str(exc)}))
        return JSONResponse(status_code=ERROR_STATUS[type(exc)], content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Full traceback goes to server logs only; the client gets a generic message.
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("Unhandled error (request_id=%s)", request_id)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/health", response_model=HealthResponse)
    async def health(llm: VLLMClient = Depends(get_llm)):
        """Readiness: the gateway is up AND vLLM can serve requests."""
        if await llm.is_ready():
            return HealthResponse(status="ok", vllm="ready")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(status="unavailable", vllm="not_ready").model_dump(),
        )

    @app.post(
        "/v1/summarize",
        response_model=SummarizeResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
        dependencies=[Depends(require_api_key)],
    )
    async def summarize(
        body: SummarizeRequest, request: Request, llm: VLLMClient = Depends(get_llm)
    ) -> SummarizeResponse | StreamingResponse:
        request.state.log_extra["input_chars"] = len(body.text)
        messages = build_messages(body.text, body.style)
        start = time.perf_counter()

        if body.stream:
            # Upstream errors raised here still produce a normal 503/504/502.
            upstream = await llm.open_stream(messages, body.max_tokens)
            return StreamingResponse(
                stream_summary(llm, upstream, request.state.request_id, start),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        completion = await llm.chat(messages, body.max_tokens)
        latency_ms = elapsed_ms(start)
        request.state.log_extra.update(
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )
        return SummarizeResponse(
            summary=completion.text,
            model=completion.model,
            usage=usage_of(completion),
            latency_ms=latency_ms,
        )

    return app
