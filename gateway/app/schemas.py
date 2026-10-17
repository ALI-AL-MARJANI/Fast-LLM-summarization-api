"""Request and response models of the public API."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Input bounds. 20k characters is roughly 5k tokens of English text, which
# leaves room for the prompt template and up to 512 output tokens inside
# vLLM's 8192-token context window (`--max-model-len`).
MIN_TEXT_CHARS = 20
MAX_TEXT_CHARS = 20_000
MIN_OUTPUT_TOKENS = 32
MAX_OUTPUT_TOKENS = 512


class SummaryStyle(str, Enum):
    SHORT = "short"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"


class SummarizeRequest(BaseModel):
    # Reject unknown fields so that client typos surface as 422 errors.
    model_config = ConfigDict(extra="forbid")

    text: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=MIN_TEXT_CHARS,
            max_length=MAX_TEXT_CHARS,
        ),
    ]
    style: SummaryStyle = SummaryStyle.SHORT
    max_tokens: int = Field(default=256, ge=MIN_OUTPUT_TOKENS, le=MAX_OUTPUT_TOKENS)
    # When true, the summary is returned incrementally as Server-Sent Events.
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    usage: Usage
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    vllm: str
