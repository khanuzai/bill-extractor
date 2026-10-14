"""Run LLM extraction on one bill PDF and print the result.

    python -m src.extraction.cli data/generated/<dev id>.pdf
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.extraction.extract import MAX_RETRIES, ExtractionResult, extract_bill

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_IDS_PATH = REPO_ROOT / "data" / "labels" / "test_ids.json"


def is_test_set_bill(pdf_path: Path, test_ids_path: Path) -> bool:
    """Whether the PDF is one of the held-out test-set bills."""
    if not test_ids_path.exists():
        return False
    return pdf_path.stem in json.loads(test_ids_path.read_text(encoding="utf-8"))


def format_summary(result: ExtractionResult) -> str:
    """A human-readable account of the attempts and the final validation report."""
    lines = [f"Retries used: {result.retries} of {MAX_RETRIES}"]
    for number, attempt in enumerate(result.attempts, start=1):
        if attempt.error is not None:
            outcome = f"no valid bill ({attempt.error.splitlines()[0]})"
        elif attempt.failed_checks:
            outcome = "failed " + ", ".join(attempt.failed_checks)
        else:
            outcome = "all checks passed"
        lines.append(f"  Attempt {number}: {outcome}")
    report = result.validation
    if report is None:
        lines.append("Validation: not run (no valid bill)")
        return "\n".join(lines)
    total = len(report.passed) + len(report.failures)
    lines.append(f"Validation: {len(report.passed)}/{total} checks passed")
    lines.extend(f"  PASS {name}" for name in report.passed)
    lines.extend(f"  FAIL {name}: {message}" for name, message in report.failures.items())
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract one bill PDF with the LLM. Prints the result as JSON on stdout and the validation report and retry count on stderr.")
    parser.add_argument("pdf", type=Path, help="path to a bill PDF (dev set only)")
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        parser.error(f"no such file: {args.pdf}")
    if is_test_set_bill(args.pdf, TEST_IDS_PATH):
        print(
            f"Refusing to run: {args.pdf.stem} is in the test set ({TEST_IDS_PATH.name}). "
            "Extraction may only be developed against dev-set bills.",
            file=sys.stderr,
        )
        return 2

    result = extract_bill(args.pdf)
    print(result.model_dump_json(indent=2))
    print(format_summary(result), file=sys.stderr)
    return 0 if result.bill is not None else 1


if __name__ == "__main__":
    sys.exit(main())
