"""Thin async client for vLLM's OpenAI-compatible API.

Network and protocol failures are translated into a small set of domain
exceptions, so the HTTP layer can map each one to a precise status code.
"""

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


class VLLMClient:
    def __init__(self, http: httpx.AsyncClient, model: str, temperature: float) -> None:
        # `http` is shared and owned by the app lifespan; never closed here.
        self._http = http
        self._model = model
        self._temperature = temperature

    async def chat(self, messages: list[dict[str, str]], max_tokens: int) -> Completion:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
        }
        try:
            response = await self._http.post("/v1/chat/completions", json=payload)
        except httpx.ConnectTimeout as exc:
            # Could not even open a connection: the server is effectively down.
            raise LLMUnavailableError("vLLM is unreachable (connect timeout).") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("vLLM did not respond in time.") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError("vLLM is unreachable.") from exc

        if response.status_code >= 500:
            raise LLMUnavailableError(f"vLLM returned HTTP {response.status_code}.")
        if response.status_code != 200:
            raise LLMResponseError(f"vLLM rejected the request (HTTP {response.status_code}).")

        try:
            data = response.json()
            return Completion(
                text=data["choices"][0]["message"]["content"].strip(),
                model=data["model"],
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMResponseError("vLLM returned an unexpected payload.") from exc

    async def is_ready(self) -> bool:
        """Readiness probe: vLLM's `/health` returns 200 once the model is loaded."""
        try:
            response = await self._http.get("/health", timeout=3.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
