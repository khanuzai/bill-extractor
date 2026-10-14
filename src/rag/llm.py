"""One structured-output call to the model, shared by the RAG steps.

Reuses the extraction client: the same model settings (EXTRACTION_MODEL etc.),
transient-error retries and usage accounting.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

import anthropic
from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from src.extraction.client import ExtractionConfig, request_extraction
from src.extraction.prompt import ExtractionPrompt

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMReply(Generic[T]):
    """The parsed reply (None if it did not match the schema) and what it cost."""

    parsed: T | None
    error: str | None
    input_tokens: int
    output_tokens: int


def ask(client: anthropic.Anthropic, config: ExtractionConfig, system: str, user: str, output: type[T]) -> LLMReply[T]:
    """Send one request whose reply must match `output`'s JSON schema.

    Never raises on a bad reply; transient API errors are retried by the client and
    re-raised if they persist.
    """
    prompt = ExtractionPrompt(
        system=system, user_message=user, output_schema=transform_schema(output.model_json_schema())
    )
    response = request_extraction(client, prompt, config)
    usage = (response.usage.input_tokens, response.usage.output_tokens)
    if response.stop_reason in ("refusal", "max_tokens") or response.payload is None:
        return LLMReply(None, f"no usable reply (stop_reason={response.stop_reason!r})", *usage)
    try:
        return LLMReply(output.model_validate(response.payload), None, *usage)
    except ValidationError as exc:
        return LLMReply(None, str(exc), *usage)
