import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.generator.cli import DEV_IDS_FILE, TEST_IDS_FILE, generate_dataset
from src.generator.data import generate_bill, generate_extras, make_faker
from src.generator.render import TEMPLATES, render_html
from src.schema import BillLabel, UtilityBill

SAMPLE_SIZE = 2000


def _bills(seed: int, count: int = SAMPLE_SIZE) -> list[UtilityBill]:
    fake = make_faker(seed)
    return [generate_bill(fake) for _ in range(count)]


@pytest.fixture(scope="module")
def bills() -> list[UtilityBill]:
    return _bills(seed=1234)


def test_bills_round_trip_through_schema(bills: list[UtilityBill]) -> None:
    for bill in bills:
        assert UtilityBill.model_validate_json(bill.model_dump_json()) == bill


def test_total_is_exact_sum_of_line_items(bills: list[UtilityBill]) -> None:
    for bill in bills:
        assert bill.total_amount == sum(item.amount for item in bill.line_items)
        assert bill.total_amount > 0
        assert 2 <= len(bill.line_items) <= 6
        for amount in [bill.total_amount, *(item.amount for item in bill.line_items)]:
            assert amount == amount.quantize(Decimal("0.01"))
            assert amount.as_tuple().exponent == -2


def test_dates_and_usage_are_consistent(bills: list[UtilityBill]) -> None:
    for bill in bills:
        assert bill.billing_period_start < bill.billing_period_end
        if bill.due_date is not None:
            assert bill.due_date > bill.billing_period_end
        assert (bill.usage_amount is None) == (bill.usage_unit is None)
        if bill.usage_amount is not None:
            assert bill.usage_amount > 0


def test_variety_matches_target_rates(bills: list[UtilityBill]) -> None:
    def share(predicate_hits: int) -> float:
        return predicate_hits / len(bills)

    previous = share(sum(any("alance" in item.description for item in bill.line_items) for bill in bills))
    credit = share(sum(any(item.amount < 0 for item in bill.line_items) for bill in bills))
    no_due = share(sum(bill.due_date is None for bill in bills))
    assert previous == pytest.approx(0.25, abs=0.04)
    assert credit == pytest.approx(0.15, abs=0.04)
    assert no_due == pytest.approx(0.10, abs=0.03)
    assert {bill.usage_unit for bill in bills} == {"kWh", "m³", None}


def test_fixed_seed_reproduces_identical_bills() -> None:
    assert _bills(seed=42, count=200) == _bills(seed=42, count=200)
    assert _bills(seed=42, count=200) != _bills(seed=43, count=200)


@pytest.mark.parametrize("template", TEMPLATES)
def test_every_template_prints_every_label_field(template: str) -> None:
    fake = make_faker(7)
    for _ in range(40):
        bill = generate_bill(fake)
        html = render_html(bill, generate_extras(fake, bill), template)
        assert bill.account_number in html
        assert bill.currency in html
        assert f"{bill.total_amount:,.2f}" in html
        for item in bill.line_items:
            assert f"{abs(item.amount):,.2f}" in html


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_cli_output_is_reproducible_and_evenly_rotated(tmp_path: Path) -> None:
    runs: list[tuple[dict[str, bytes], dict[str, bytes]]] = []
    for name in ("first", "second"):
        pdf_dir, label_dir = tmp_path / name / "generated", tmp_path / name / "labels"
        generate_dataset(count=20, seed=42, split=True, pdf_dir=pdf_dir, label_dir=label_dir)
        runs.append((_files(pdf_dir), _files(label_dir)))
    assert runs[0] == runs[1]

    pdfs, label_files = runs[0]
    dev_ids = json.loads(label_files.pop(DEV_IDS_FILE))
    test_ids = json.loads(label_files.pop(TEST_IDS_FILE))
    labels = [BillLabel.model_validate_json(content) for content in label_files.values()]

    assert len(pdfs) == len(labels) == 20
    assert {f"{label.id}.pdf" for label in labels} == set(pdfs)
    assert all(content.startswith(b"%PDF") for content in pdfs.values())
    for template in TEMPLATES:
        assert sum(label.template == template for label in labels) == 4

    assert len(dev_ids) == len(test_ids) == 10
    assert set(dev_ids).isdisjoint(test_ids)
    assert set(dev_ids) | set(test_ids) == {label.id for label in labels}


def test_rerun_with_smaller_count_leaves_no_stale_bills(tmp_path: Path) -> None:
    pdf_dir, label_dir = tmp_path / "generated", tmp_path / "labels"
    generate_dataset(count=10, seed=1, split=False, pdf_dir=pdf_dir, label_dir=label_dir)
    generate_dataset(count=5, seed=1, split=False, pdf_dir=pdf_dir, label_dir=label_dir)
    assert len(list(pdf_dir.iterdir())) == len(list(label_dir.iterdir())) == 5
