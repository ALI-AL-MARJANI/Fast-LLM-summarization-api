"""Gateway configuration, loaded from environment variables (and `.env`)."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated settings. Invalid or missing values fail fast at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # Shared secret expected in the `X-API-Key` header.
    api_key: SecretStr = Field(min_length=16)

    # Upstream vLLM server (OpenAI-compatible API).
    vllm_base_url: str = "http://vllm:8000"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Connect timeout is short: if vLLM is unreachable we want to know quickly.
    # Request timeout bounds the full generation time of one summary.
    vllm_connect_timeout_s: float = Field(default=5.0, gt=0)
    vllm_request_timeout_s: float = Field(default=60.0, gt=0)

    # Low temperature keeps summaries faithful and stable across calls.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    log_level: str = "INFO"
