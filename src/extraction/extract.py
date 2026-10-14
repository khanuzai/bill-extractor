"""Extract a `UtilityBill` from a bill PDF with the LLM, retrying on failed checks."""

from dataclasses import dataclass
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
from src.extraction.prompt import ExtractionPrompt, build_prompt, build_retry_prompt
from src.extraction.validate import ValidationReport, validate_bill
from src.parsers.text import extract_text
from src.schema import UtilityBill

MAX_RETRIES = 2

# Stop reasons that feedback cannot fix, so they are not retried.
TERMINAL_STOP_REASONS = ("refusal", "max_tokens")


class AttemptRecord(BaseModel):
    """What happened on one request to the model."""

    failed_checks: list[str]
    """Checks the attempt's bill failed; empty if it passed them all or had no bill."""

    error: str | None
    """Why the attempt produced no bill (bad output or API error); None if it did."""


class ExtractionResult(BaseModel):
    """The outcome of extracting one bill, successful or not."""

    bill: UtilityBill | None
    """The best attempt's bill: the one failing the fewest checks, the later one on a
    tie. It may still fail checks (see `validation`). None if no attempt produced a
    valid `UtilityBill`."""

    error: str | None
    """Why `bill` is None: the schema validation message, or a description of the
    failed response or API error. None whenever there is a bill."""

    validation: ValidationReport | None
    """The checks run on `bill`; None when there is no bill."""

    retries: int
    """Follow-up requests sent after the first attempt (0 to MAX_RETRIES)."""

    attempts: list[AttemptRecord]
    """One record per request, in order: which checks failed, or why there was no bill."""

    input_tokens: int
    output_tokens: int
    latency_seconds: float
    """Totals over every attempt's API request; zero if no request completed."""

    model: str
    """The model that produced the output."""

    raw_output: str | None
    """The best attempt's unparsed response text, kept for debugging failures."""


@dataclass(frozen=True)
class _Attempt:
    bill: UtilityBill | None
    report: ValidationReport | None
    error: str | None
    raw_output: str

    def rank(self) -> tuple[bool, int]:
        """Higher is better: having a bill first, then fewer failed checks."""
        if self.bill is None or self.report is None:
            return (False, 0)
        return (True, -len(self.report.failures))


def _parse(response: ModelResponse) -> tuple[UtilityBill | None, str | None]:
    if response.stop_reason in TERMINAL_STOP_REASONS:
        return None, f"model stopped with stop_reason={response.stop_reason!r} before completing the output"
    if response.payload is None:
        return None, "model output is not a JSON object"
    try:
        return UtilityBill.model_validate(response.payload), None
    except ValidationError as exc:
        return None, str(exc)


def _evaluate(response: ModelResponse) -> _Attempt:
    bill, error = _parse(response)
    report = validate_bill(bill) if bill is not None else None
    return _Attempt(bill=bill, report=report, error=error, raw_output=response.raw_text)


def _feedback(attempt: _Attempt) -> list[str]:
    if attempt.report is not None:
        return list(attempt.report.failures.values())
    return [f"your output was not a valid bill: {attempt.error}"]


def extract_bill(
    pdf_path: Path,
    client: anthropic.Anthropic | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Extract the bill in `pdf_path`, giving the model up to MAX_RETRIES corrections.

    Each attempt is checked with `validate_bill`. If a check fails (or the output is
    not a valid bill), the model is sent the bill text, its previous JSON and the
    failure messages, and asked again. It stops as soon as every check passes. A bill
    that still fails after the last retry is returned as its best attempt with the
    failing report, not discarded.

    Never raises on a bad model response. A transient API error that outlasts the
    client's retries ends the loop and is recorded. Errors that mean the setup is wrong
    (unreadable PDF, bad API key, rejected request) do raise.
    """
    config = config or load_config()
    client = client or make_client()
    base_prompt = build_prompt(extract_text(pdf_path))

    prompt: ExtractionPrompt = base_prompt
    records: list[AttemptRecord] = []
    best: _Attempt | None = None
    input_tokens = output_tokens = 0
    latency = 0.0
    model = config.model
    for attempt_number in range(MAX_RETRIES + 1):
        try:
            response = request_extraction(client, prompt, config)
        except TRANSIENT_ERRORS as exc:
            records.append(AttemptRecord(failed_checks=[], error=f"API error after retries: {type(exc).__name__}: {exc}"))
            break
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        latency += response.usage.latency_seconds
        model = response.model

        attempt = _evaluate(response)
        records.append(
            AttemptRecord(failed_checks=attempt.report.failed if attempt.report else [], error=attempt.error)
        )
        if best is None or attempt.rank() >= best.rank():
            best = attempt
        if attempt.report is not None and attempt.report.all_passed:
            break
        if response.stop_reason in TERMINAL_STOP_REASONS or attempt_number == MAX_RETRIES:
            break
        prompt = build_retry_prompt(base_prompt, response.raw_text, _feedback(attempt))

    bill = best.bill if best else None
    return ExtractionResult(
        bill=bill,
        error=None if bill is not None else (best.error if best else records[-1].error),
        validation=best.report if best else None,
        retries=len(records) - 1,
        attempts=records,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        model=model,
        raw_output=best.raw_output if best else None,
    )
