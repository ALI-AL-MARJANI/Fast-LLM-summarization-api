"""Thin async client for vLLM's OpenAI-compatible API.

Network and protocol failures are translated into a small set of domain
exceptions, so the HTTP layer can map each one to a precise status code.
"""

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx


class LLMError(Exception):
    """Base class for upstream model-server failures."""


class LLMUnavailableError(LLMError):
    """vLLM is unreachable or returned a server error (-> 503)."""


class LLMTimeoutError(LLMError):
    """vLLM accepted the request but did not answer in time (-> 504)."""


class LLMResponseError(LLMError):
    """vLLM rejected the request or returned an unexpected payload (-> 502)."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@contextmanager
def _translate_errors() -> Iterator[None]:
    """Map httpx and parsing exceptions to domain exceptions."""
    try:
        yield
    except httpx.ConnectTimeout as exc:
        # Could not even open a connection: the server is effectively down.
        raise LLMUnavailableError("vLLM is unreachable (connect timeout).") from exc
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError("vLLM did not respond in time.") from exc
    except httpx.TransportError as exc:
        raise LLMUnavailableError("vLLM is unreachable.") from exc
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMResponseError("vLLM returned an unexpected payload.") from exc


def _raise_for_status(status_code: int) -> None:
    if status_code >= 500:
        raise LLMUnavailableError(f"vLLM returned HTTP {status_code}.")
    if status_code != 200:
        raise LLMResponseError(f"vLLM rejected the request (HTTP {status_code}).")


class VLLMClient:
    def __init__(self, http: httpx.AsyncClient, model: str, temperature: float) -> None:
        # `http` is shared and owned by the app lifespan; never closed here.
        self._http = http
        self._model = model
        self._temperature = temperature

    def _payload(self, messages: list[dict[str, str]], max_tokens: int) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
        }

    async def chat(self, messages: list[dict[str, str]], max_tokens: int) -> Completion:
        with _translate_errors():
            response = await self._http.post(
                "/v1/chat/completions", json=self._payload(messages, max_tokens)
            )
            _raise_for_status(response.status_code)
            data = response.json()
            return Completion(
                text=data["choices"][0]["message"]["content"].strip(),
                model=data["model"],
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
            )

    async def open_stream(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> httpx.Response:
        """Start a streamed completion and return once vLLM has sent its headers.

        Connection and HTTP errors are raised here, *before* the gateway starts
        its own response, so they still map to a proper 503/504/502 status.
        """
        payload = self._payload(messages, max_tokens) | {
            "stream": True,
            # Ask vLLM for a final chunk carrying token counts.
            "stream_options": {"include_usage": True},
        }
        request = self._http.build_request("POST", "/v1/chat/completions", json=payload)
        with _translate_errors():
            response = await self._http.send(request, stream=True)
        if response.status_code != 200:
            await response.aclose()
            _raise_for_status(response.status_code)
        return response

    async def iter_stream(self, response: httpx.Response) -> AsyncIterator[str | Completion]:
        """Yield text deltas as they arrive, then one final `Completion` with usage.

        vLLM sends Server-Sent Events: `data: {json chunk}` lines, then `data: [DONE]`.
        """
        parts: list[str] = []
        model, usage = self._model, None
        try:
            with _translate_errors():
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue  # blank separator lines between events
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    model = chunk.get("model", model)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        if delta := choice.get("delta", {}).get("content"):
                            parts.append(delta)
                            yield delta
                if usage is None:
                    raise LLMResponseError("vLLM stream ended without usage information.")
                yield Completion(
                    text="".join(parts).strip(),
                    model=model,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                )
        finally:
            # Release the pooled connection, also when the client disconnects early.
            await response.aclose()

    async def is_ready(self) -> bool:
        """Readiness probe: vLLM's `/health` returns 200 once the model is loaded."""
        try:
            response = await self._http.get("/health", timeout=3.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
