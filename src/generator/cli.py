"""Generate a labeled set of synthetic bill PDFs.

    python -m src.generator.cli --count 200 --seed 42 --split
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.generator.data import generate_bill, generate_extras, make_faker
from src.generator.render import TEMPLATES, render_pdf
from src.schema import BillLabel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIR = REPO_ROOT / "data" / "generated"
DEFAULT_LABEL_DIR = REPO_ROOT / "data" / "labels"

ID_PREFIX = "bill_"
DEV_IDS_FILE = "dev_ids.json"
TEST_IDS_FILE = "test_ids.json"


def _clear_previous_run(pdf_dir: Path, label_dir: Path) -> None:
    """Delete bills written by an earlier run so a smaller run leaves no stale files."""
    for path in [*pdf_dir.glob(f"{ID_PREFIX}*.pdf"), *label_dir.glob(f"{ID_PREFIX}*.json")]:
        path.unlink()


def split_ids(labels: Sequence[BillLabel], seed: int | None) -> tuple[list[str], list[str]]:
    """Split ids 50/50 into (dev, test), balanced across templates."""
    rng = make_faker(seed).random
    shuffled: list[str] = []
    for template in TEMPLATES:
        ids = [label.id for label in labels if label.template == template]
        rng.shuffle(ids)
        shuffled.extend(ids)
    return sorted(shuffled[0::2]), sorted(shuffled[1::2])


def generate_dataset(
    count: int,
    seed: int | None,
    split: bool,
    pdf_dir: Path = DEFAULT_PDF_DIR,
    label_dir: Path = DEFAULT_LABEL_DIR,
) -> list[BillLabel]:
    """Write `count` bill PDFs and their labels, rotating evenly through the templates."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_run(pdf_dir, label_dir)

    fake = make_faker(seed)
    width = max(4, len(str(count)))
    labels: list[BillLabel] = []
    for index in range(count):
        bill = generate_bill(fake)
        extras = generate_extras(fake, bill)
        label = BillLabel(
            id=f"{ID_PREFIX}{index + 1:0{width}d}",
            template=TEMPLATES[index % len(TEMPLATES)],
            bill=bill,
        )
        render_pdf(bill, extras, label.template, pdf_dir / f"{label.id}.pdf")
        (label_dir / f"{label.id}.json").write_text(label.model_dump_json(indent=2), encoding="utf-8")
        labels.append(label)

    if split:
        dev_ids, test_ids = split_ids(labels, seed)
        (label_dir / DEV_IDS_FILE).write_text(json.dumps(dev_ids, indent=2), encoding="utf-8")
        (label_dir / TEST_IDS_FILE).write_text(json.dumps(test_ids, indent=2), encoding="utf-8")
    return labels


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic utility bill PDFs with ground-truth labels.")
    parser.add_argument("--count", type=int, default=200, help="number of bills to generate (default: 200)")
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible output (default: random)")
    parser.add_argument("--split", action="store_true", help=f"also write {DEV_IDS_FILE} and {TEST_IDS_FILE} (50/50)")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="where to write PDFs")
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="where to write labels")
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")

    labels = generate_dataset(args.count, args.seed, args.split, args.pdf_dir, args.label_dir)
    print(f"Wrote {len(labels)} PDFs to {args.pdf_dir} and labels to {args.label_dir}")


if __name__ == "__main__":
    main()
