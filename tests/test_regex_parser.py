from decimal import Decimal
from pathlib import Path

import pytest

from src.generator.data import generate_bill, generate_extras, make_faker
from src.generator.render import TEMPLATES, render_pdf
from src.parsers.regex_parser import ParseResult, parse_bill_text, parse_date, parse_money
from src.parsers.text import extract_text
from src.schema import UtilityBill

TARGET_TEMPLATES = ("bordered_table", "two_column")
UNSEEN_TEMPLATES = ("summary_box", "two_page", "receipt")
BILLS_PER_TEMPLATE = 40
BILL_FIELDS = tuple(UtilityBill.model_fields)

Sample = tuple[UtilityBill, ParseResult]


@pytest.fixture(scope="module")
def parsed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[Sample]]:
    """Every template -> the same bills, rendered to PDF, extracted and parsed."""
    fake = make_faker(2024)
    bills = [generate_bill(fake) for _ in range(BILLS_PER_TEMPLATE)]
    extras = [generate_extras(fake, bill) for bill in bills]
    directory = tmp_path_factory.mktemp("pdfs")
    samples: dict[str, list[Sample]] = {}
    for template in TEMPLATES:
        samples[template] = []
        for index, (bill, extra) in enumerate(zip(bills, extras)):
            pdf_path = directory / f"{template}_{index}.pdf"
            render_pdf(bill, extra, template, pdf_path)
            samples[template].append((bill, parse_bill_text(extract_text(pdf_path))))
    return samples


def test_templates_are_all_classified() -> None:
    assert set(TARGET_TEMPLATES) | set(UNSEEN_TEMPLATES) == set(TEMPLATES)


def test_sample_covers_the_hard_cases(parsed: dict[str, list[Sample]]) -> None:
    bills = [bill for bill, _ in parsed[TARGET_TEMPLATES[0]]]
    assert any(bill.due_date is None for bill in bills)
    assert any(bill.usage_amount is None for bill in bills)
    assert any(item.amount < 0 for bill in bills for item in bill.line_items)


@pytest.mark.parametrize("template", TARGET_TEMPLATES)
def test_target_templates_parse_every_field(template: str, parsed: dict[str, list[Sample]]) -> None:
    for bill, result in parsed[template]:
        for field in BILL_FIELDS:
            assert getattr(result, field) == getattr(bill, field), field
        assert result.found == [field for field in BILL_FIELDS if getattr(bill, field) is not None]


@pytest.mark.parametrize("template", UNSEEN_TEMPLATES)
def test_unseen_templates_yield_none_not_wrong_values(template: str, parsed: dict[str, list[Sample]]) -> None:
    for bill, result in parsed[template]:
        for field in result.found:
            assert getattr(result, field) == getattr(bill, field), field
        assert len(result.found) < len(BILL_FIELDS) / 2


@pytest.mark.parametrize(
    "text",
    ["", "\n\n", "lorem ipsum dolor sit amet", "Account Number\n\nAmount Due\nTotal Balance  soon USD", "Due Date  99/99/2026"],
)
def test_unrecognized_text_never_raises(text: str) -> None:
    result = parse_bill_text(text)
    assert result.found == []
    assert result.model_dump(exclude={"found"}) == dict.fromkeys(BILL_FIELDS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("-7.33", Decimal("-7.33")),
        ("(87.74)", Decimal("-87.74")),
        ("26.44 CR", Decimal("-26.44")),
        ("$(87.74)", Decimal("-87.74")),
        ("-1,234.50", Decimal("-1234.50")),
        ("(1,234.50)", Decimal("-1234.50")),
        ("1,234.50 CR", Decimal("-1234.50")),
        ("7.33", Decimal("7.33")),
        ("$1,234.50", Decimal("1234.50")),
    ],
)
def test_parse_money_returns_signed_decimal(text: str, expected: Decimal) -> None:
    assert parse_money(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "12", "12.5", "(12.50", "1,23.45", "12.50 DR", "12.50 apples"])
def test_parse_money_rejects_non_amounts(text: str) -> None:
    assert parse_money(text) is None


def test_parse_date_handles_target_formats_only() -> None:
    assert parse_date("03/15/2026").isoformat() == "2026-03-15"
    assert parse_date("March 5, 2026").isoformat() == "2026-03-05"
    assert parse_date("15-Mar-2026") is None
    assert parse_date("2026-03-15") is None
    assert parse_date("02/30/2026") is None
    assert parse_date("Upon receipt") is None
