"""Extract a `UtilityBill` from a bill PDF with the LLM."""

from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

from src.extraction.client import (
    TRANSIENT_ERRORS,
    ExtractionConfig,
    ModelResponse,
    load_config,
    make_client,
    request_extraction,
)
from src.extraction.prompt import build_prompt
from src.parsers.text import extract_text
from src.schema import UtilityBill


class ExtractionResult(BaseModel):
    """The outcome of extracting one bill, successful or not."""

    bill: UtilityBill | None
    """The extracted bill; None if the API call failed or the output was not a valid bill."""

    error: str | None
    """Why `bill` is None: the schema validation message, or a description of the
    failed response or API error. None on success."""

    input_tokens: int
    output_tokens: int
    latency_seconds: float
    """Token usage and wall-clock time of the API request; zero if no request completed."""

    model: str
    """The model that produced the output."""

    raw_output: str | None
    """The model's unparsed response text, kept for debugging failures."""


def _validate(response: ModelResponse) -> tuple[UtilityBill | None, str | None]:
    if response.stop_reason in ("refusal", "max_tokens"):
        return None, f"model stopped with stop_reason={response.stop_reason!r} before completing the output"
    if response.payload is None:
        return None, "model output is not a JSON object"
    try:
        return UtilityBill.model_validate(response.payload), None
    except ValidationError as exc:
        return None, str(exc)


def extract_bill(
    pdf_path: Path,
    client: anthropic.Anthropic | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Extract the bill in `pdf_path`.

    Never raises on a bad model response: output that is refused, truncated, not JSON
    or not a valid `UtilityBill` comes back with `bill=None` and the reason in `error`.
    The same goes for a transient API error that outlasts the retries. Errors that
    mean the setup is wrong (unreadable PDF, bad API key, rejected request) do raise.
    """
    config = config or load_config()
    client = client or make_client()
    prompt = build_prompt(extract_text(pdf_path))
    try:
        response = request_extraction(client, prompt, config)
    except TRANSIENT_ERRORS as exc:
        return ExtractionResult(
            bill=None,
            error=f"API error after retries: {type(exc).__name__}: {exc}",
            input_tokens=0,
            output_tokens=0,
            latency_seconds=0.0,
            model=config.model,
            raw_output=None,
        )
    bill, error = _validate(response)
    return ExtractionResult(
        bill=bill,
        error=error,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_seconds=response.usage.latency_seconds,
        model=response.model,
        raw_output=response.raw_text,
    )
