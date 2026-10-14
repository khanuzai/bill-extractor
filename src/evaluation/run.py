"""Score the regex baseline and the LLM pipeline on the test set.

    python -m src.evaluation.run --limit 5   # smoke run: writes smoke_results.*
    python -m src.evaluation.run             # full run: writes results.*

Runs on the bills in data/labels/test_ids.json only. LLM results are cached by bill id
in data/results/llm_cache.json, so a rerun (or the full run after a smoke run) does not
pay for bills already extracted. The test set is scored once: do not tune prompts on
these results.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

import anthropic
from pydantic import BaseModel

from src.evaluation.cost import cost_usd
from src.evaluation.records import BillRecord, LLMRecord, RegexRecord, RunResults
from src.evaluation.report import render_report
from src.evaluation.score import score_bill
from src.extraction.client import load_config, make_client
from src.extraction.extract import ExtractionResult, extract_bill
from src.parsers.regex_parser import parse_bill_text
from src.parsers.text import extract_text
from src.schema import BillLabel

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = REPO_ROOT / "data" / "labels"
PDF_DIR = REPO_ROOT / "data" / "generated"
RESULTS_DIR = REPO_ROOT / "data" / "results"
TEST_IDS_FILE = "test_ids.json"
CACHE_FILE = "llm_cache.json"

ExtractFn = Callable[[Path], ExtractionResult]


class LLMCache(BaseModel):
    """LLM extraction results keyed by bill id, for one extraction model."""

    model: str
    results: dict[str, ExtractionResult]


def load_test_ids(label_dir: Path) -> list[str]:
    path = label_dir / TEST_IDS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; generate the data with --split first")
    return sorted(json.loads(path.read_text(encoding="utf-8")))


def load_cache(path: Path, model: str) -> LLMCache:
    """Load the cache, refusing one built with a different model so results never mix."""
    if not path.exists():
        return LLMCache(model=model, results={})
    cache = LLMCache.model_validate_json(path.read_text(encoding="utf-8"))
    if cache.model != model:
        raise ValueError(
            f"{path} holds results for model {cache.model!r}, but extraction is configured for {model!r}. "
            "Pass a different --cache path to evaluate another model."
        )
    return cache


def save_cache(path: Path, cache: LLMCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cache.model_dump_json(indent=2), encoding="utf-8")


def _run_regex(pdf_path: Path, label: BillLabel) -> RegexRecord:
    started = time.perf_counter()
    result = parse_bill_text(extract_text(pdf_path))
    latency = time.perf_counter() - started
    return RegexRecord(score=score_bill(result, label.bill), found=result.found, latency_seconds=latency)


def _llm_record(result: ExtractionResult, label: BillLabel, cached: bool) -> LLMRecord:
    return LLMRecord(
        score=score_bill(result.bill, label.bill),
        error=result.error,
        validation=result.validation,
        retries=result.retries,
        attempts=result.attempts,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_seconds=result.latency_seconds,
        model=result.model,
        cost_usd=cost_usd(result.model, result.input_tokens, result.output_tokens),
        cached=cached,
    )


def evaluate(
    ids: Sequence[str],
    label_dir: Path,
    pdf_dir: Path,
    extract: ExtractFn,
    cache_path: Path,
    model: str,
    limit: int | None = None,
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> RunResults:
    """Run both approaches on `ids`, extracting with `extract` only for uncached bills.

    The cache is saved after every new extraction, so an interrupted run loses nothing
    it already paid for.
    """
    cache = load_cache(cache_path, model)
    bills: list[BillRecord] = []
    new_calls = 0
    cost_this_run = cost_total = Decimal(0)
    unpriced = 0
    for number, bill_id in enumerate(ids, start=1):
        label = BillLabel.model_validate_json((label_dir / f"{bill_id}.json").read_text(encoding="utf-8"))
        pdf_path = pdf_dir / f"{bill_id}.pdf"

        cached = bill_id in cache.results
        if not cached:
            cache.results[bill_id] = extract(pdf_path)
            save_cache(cache_path, cache)
            new_calls += 1
        llm = _llm_record(cache.results[bill_id], label, cached)
        if llm.cost_usd is None:
            unpriced += 1
        else:
            cost_total += llm.cost_usd
            if not cached:
                cost_this_run += llm.cost_usd

        bills.append(BillRecord(id=bill_id, template=label.template, regex=_run_regex(pdf_path, label), llm=llm))
        log(f"[{number}/{len(ids)}] {bill_id} {label.template}: LLM {'cached' if cached else 'extracted'}")

    return RunResults(
        model=model,
        limit=limit,
        bills=bills,
        new_api_calls=new_calls,
        cost_this_run_usd=cost_this_run,
        cost_total_usd=cost_total,
        unpriced_bills=unpriced,
    )


def write_results(results: RunResults, results_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write the raw results as JSON and the report as markdown; return both paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{stem}.json"
    markdown_path = results_dir / f"{stem}.md"
    json_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_report(results), encoding="utf-8")
    return json_path, markdown_path


def _lazy_extractor() -> ExtractFn:
    """Extract with the configured client, built only when a bill is not cached."""
    config = load_config()
    client_holder: list[anthropic.Anthropic] = []

    def extract(pdf_path: Path) -> ExtractionResult:
        if not client_holder:
            client_holder.append(make_client())
        return extract_bill(pdf_path, client=client_holder[0], config=config)

    return extract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the regex baseline and the LLM pipeline on the test set.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N test bills (smoke run)")
    parser.add_argument("--cache", type=Path, default=RESULTS_DIR / CACHE_FILE, help="LLM result cache")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="where to write results")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    ids = load_test_ids(LABEL_DIR)
    if args.limit is not None:
        ids = ids[: args.limit]
    model = load_config().model
    results = evaluate(ids, LABEL_DIR, PDF_DIR, _lazy_extractor(), args.cache, model, args.limit)
    json_path, markdown_path = write_results(results, args.results_dir, "smoke_results" if args.limit else "results")

    print(f"Scored {len(results.bills)} test bills with {model}.")
    print(f"Wrote {json_path} and {markdown_path}")
    print(f"API cost this run: ${results.cost_this_run_usd:.4f} ({results.new_api_calls} bills extracted)")
    print(f"API cost of all {len(results.bills)} LLM results, cached included: ${results.cost_total_usd:.4f}")
    if results.unpriced_bills:
        print(f"Warning: {results.unpriced_bills} bills have no cost; their model is not in the price table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
