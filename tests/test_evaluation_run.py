"""End-to-end check of the evaluation harness on a throwaway dataset.

Bills are generated into a temp directory with their own seed; the real test set in
data/ is never touched, and the LLM is a fake that returns the label.
"""

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from src.evaluation.run import LLMCache, evaluate, load_test_ids, write_results
from src.extraction.extract import AttemptRecord, ExtractionResult
from src.extraction.validate import validate_bill
from src.generator.cli import generate_dataset
from src.schema import BillLabel

MODEL = "claude-haiku-4-5"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("eval")
    generate_dataset(count=10, seed=99, split=True, pdf_dir=root / "generated", label_dir=root / "labels")
    return root / "generated", root / "labels"


def _fake_extractor(label_dir: Path, calls: list[Path]) -> Callable[[Path], ExtractionResult]:
    """Returns the label as the extraction, at a fixed token cost per bill."""

    def extract(pdf_path: Path) -> ExtractionResult:
        calls.append(pdf_path)
        bill = BillLabel.model_validate_json((label_dir / f"{pdf_path.stem}.json").read_text()).bill
        return ExtractionResult(
            bill=bill,
            error=None,
            validation=validate_bill(bill),
            retries=0,
            attempts=[AttemptRecord(failed_checks=[], error=None)],
            input_tokens=2000,
            output_tokens=200,
            latency_seconds=1.5,
            model=f"{MODEL}-20251001",
            raw_output=bill.model_dump_json(),
        )

    return extract


def test_evaluate_scores_both_approaches_and_caches_llm_results(dataset: tuple[Path, Path], tmp_path: Path) -> None:
    pdf_dir, label_dir = dataset
    ids = load_test_ids(label_dir)
    cache_path = tmp_path / "llm_cache.json"
    calls: list[Path] = []
    extract = _fake_extractor(label_dir, calls)

    smoke = evaluate(ids[:2], label_dir, pdf_dir, extract, cache_path, MODEL, limit=2, log=lambda _: None)
    full = evaluate(ids, label_dir, pdf_dir, extract, cache_path, MODEL, log=lambda _: None)

    assert len(calls) == len(ids) == 5  # the two smoke bills were not re-extracted
    assert smoke.new_api_calls == 2 and full.new_api_calls == 3
    assert [bill.llm.cached for bill in full.bills] == [True, True, False, False, False]
    per_bill = Decimal("0.003")  # 2000 * $1/M + 200 * $5/M
    assert full.cost_this_run_usd == 3 * per_bill
    assert full.cost_total_usd == 5 * per_bill
    assert set(LLMCache.model_validate_json(cache_path.read_text()).results) == set(ids)

    assert all(bill.llm.score.all_correct for bill in full.bills)
    for bill in full.bills:
        regex_found = set(bill.regex.found)
        expected_found = bool(regex_found) if bill.template in ("bordered_table", "two_column") else not regex_found
        assert expected_found, bill.template


def test_cache_from_another_model_is_refused(dataset: tuple[Path, Path], tmp_path: Path) -> None:
    pdf_dir, label_dir = dataset
    cache_path = tmp_path / "llm_cache.json"
    cache_path.write_text(LLMCache(model="claude-opus-5-5", results={}).model_dump_json())
    with pytest.raises(ValueError, match="claude-opus-5-5"):
        evaluate(load_test_ids(label_dir), label_dir, pdf_dir, _fake_extractor(label_dir, []), cache_path, MODEL)


def test_results_are_written_as_json_and_markdown(dataset: tuple[Path, Path], tmp_path: Path) -> None:
    pdf_dir, label_dir = dataset
    results = evaluate(
        load_test_ids(label_dir), label_dir, pdf_dir, _fake_extractor(label_dir, []),
        tmp_path / "cache.json", MODEL, log=lambda _: None,
    )

    json_path, markdown_path = write_results(results, tmp_path / "results", "results")

    raw = json.loads(json_path.read_text())
    assert raw["model"] == MODEL and len(raw["bills"]) == 5
    report = markdown_path.read_text()
    for heading in (
        "## Overall",
        "## Per template",
        "## Per field",
        "## Line items, item level",
        "## Cost and latency",
        "## Validation retries",
        "## Validation checks fired",
    ):
        assert heading in report
    assert "**unseen**" in report
    assert "| LLM pipeline | 50 (100.0%) | 0 (0.0%) | 0 (0.0%) | 5/5 (100.0%) |" in report
