"""The data an evaluation run records, shared by run.py (writes) and report.py (reads)."""

from decimal import Decimal

from pydantic import BaseModel

from src.evaluation.score import BillScore
from src.extraction.extract import AttemptRecord
from src.extraction.validate import ValidationReport


class RegexRecord(BaseModel):
    """The baseline parser's result on one bill."""

    score: BillScore
    found: list[str]
    latency_seconds: float
    """Text extraction plus parsing, measured on this run."""


class LLMRecord(BaseModel):
    """The LLM pipeline's result on one bill."""

    score: BillScore
    error: str | None
    validation: ValidationReport | None
    retries: int
    attempts: list[AttemptRecord]
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    """API time summed over all attempts, as measured when the bill was extracted."""
    model: str
    cost_usd: Decimal | None
    """None if the model's prices are unknown."""
    cached: bool
    """True if the result came from the cache rather than a new API call on this run."""


class BillRecord(BaseModel):
    """Both approaches' results on one test-set bill."""

    id: str
    template: str
    regex: RegexRecord
    llm: LLMRecord


class RunResults(BaseModel):
    """Everything one evaluation run produced."""

    model: str
    """The extraction model the run was configured with."""
    limit: int | None
    bills: list[BillRecord]
    new_api_calls: int
    """Bills extracted with new API calls on this run (the rest came from the cache)."""
    cost_this_run_usd: Decimal
    """Cost of the bills extracted on this run."""
    cost_total_usd: Decimal
    """Cost of every LLM result in this run, cached ones included."""
    unpriced_bills: int
    """Bills whose cost is unknown because the model is not in the price table."""
