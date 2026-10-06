"""Run LLM extraction on one bill PDF and print the result.

    python -m src.extraction.cli data/generated/<dev id>.pdf
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.extraction.extract import extract_bill

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_IDS_PATH = REPO_ROOT / "data" / "labels" / "test_ids.json"


def is_test_set_bill(pdf_path: Path, test_ids_path: Path) -> bool:
    """Whether the PDF is one of the held-out test-set bills."""
    if not test_ids_path.exists():
        return False
    return pdf_path.stem in json.loads(test_ids_path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract one bill PDF with the LLM and print the result as JSON.")
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
    return 0 if result.bill is not None else 1


if __name__ == "__main__":
    sys.exit(main())
