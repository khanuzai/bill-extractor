"""Unit tests for the label-free consistency checks. Hand-built bills, no API."""

from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest

from src.extraction.validate import (
    CHECKS,
    ValidationReport,
    check_amount_precision,
    check_billing_period_order,
    check_due_date_window,
    check_line_items_present,
    check_line_items_sum,
    check_total_positive,
    validate_bill,
)
from src.schema import LineItem, UtilityBill

PERIOD_END = date(2026, 3, 10)


def make_bill(**overrides: Any) -> UtilityBill:
    """A valid electricity bill with a previous balance and a credit; override any field."""
    fields: dict[str, Any] = {
        "provider": "Lake Selena Power & Light",
        "account_number": "4098-366-5528",
        "billing_period_start": date(2026, 2, 8),
        "billing_period_end": PERIOD_END,
        "due_date": date(2026, 3, 30),
        "usage_amount": Decimal("412"),
        "usage_unit": "kWh",
        "line_items": [
            LineItem(description="Previous balance", amount=Decimal("48.49")),
            LineItem(description="Energy charge", amount=Decimal("95.12")),
            LineItem(description="Delivery charge", amount=Decimal("13.11")),
            LineItem(description="Autopay discount", amount=Decimal("-10.00")),
            LineItem(description="State tax", amount=Decimal("5.48")),
        ],
        "total_amount": Decimal("152.20"),
        "currency": "USD",
    }
    return UtilityBill(**(fields | overrides))


VALID = make_bill()


# --- the three named scenarios ----------------------------------------------


def test_valid_bill_passes_every_check() -> None:
    report = validate_bill(VALID)
    assert report.all_passed
    assert report.passed == list(CHECKS)
    assert report.failures == {}
    assert report.failed == []


def test_dropped_line_item_fails_the_sum_check_with_both_numbers() -> None:
    dropped = make_bill(line_items=[item for item in VALID.line_items if item.description != "Delivery charge"])

    report = validate_bill(dropped)

    assert report.failed == ["line_items_sum"]
    assert report.failures["line_items_sum"] == "line items sum to 139.09 but total_amount is 152.20 (difference 13.11)"


def test_credit_with_wrong_sign_fails_the_sum_check_by_twice_the_credit() -> None:
    sign_error = make_bill(
        line_items=[
            LineItem(description=item.description, amount=abs(item.amount)) for item in VALID.line_items
        ]
    )

    report = validate_bill(sign_error)

    assert report.failed == ["line_items_sum"]
    assert report.failures["line_items_sum"] == "line items sum to 172.20 but total_amount is 152.20 (difference -20.00)"


# --- line_items_sum ---------------------------------------------------------


def test_line_items_sum_passes_when_exact() -> None:
    assert check_line_items_sum(VALID) is None


def test_line_items_sum_compares_exactly_not_approximately() -> None:
    assert check_line_items_sum(make_bill(total_amount=Decimal("152.21"))) == (
        "line items sum to 152.20 but total_amount is 152.21 (difference 0.01)"
    )


def test_line_items_sum_ignores_trailing_zeros() -> None:
    assert check_line_items_sum(make_bill(total_amount=Decimal("152.2"))) is None


# --- billing_period_order ---------------------------------------------------


def test_billing_period_order_passes_when_end_is_after_start() -> None:
    assert check_billing_period_order(VALID) is None


@pytest.mark.parametrize("end", [date(2026, 2, 8), date(2026, 1, 9)], ids=["same day", "before start"])
def test_billing_period_order_fails_unless_end_is_strictly_after_start(end: date) -> None:
    bill = make_bill(billing_period_end=end, due_date=None)
    assert check_billing_period_order(bill) == f"billing_period_end {end} is not after billing_period_start 2026-02-08"


# --- due_date_window --------------------------------------------------------


@pytest.mark.parametrize("days_after", [0, 20, 90])
def test_due_date_window_passes_from_0_to_90_days_after_period_end(days_after: int) -> None:
    assert check_due_date_window(make_bill(due_date=PERIOD_END + timedelta(days=days_after))) is None


def test_due_date_window_passes_when_there_is_no_due_date() -> None:
    assert check_due_date_window(make_bill(due_date=None)) is None


def test_due_date_window_fails_before_period_end() -> None:
    bill = make_bill(due_date=date(2026, 3, 5))
    assert check_due_date_window(bill) == "due_date 2026-03-05 is 5 days before billing_period_end 2026-03-10"


def test_due_date_window_fails_more_than_90_days_after_period_end() -> None:
    bill = make_bill(due_date=PERIOD_END + timedelta(days=91))
    assert check_due_date_window(bill) == (
        "due_date 2026-06-09 is 91 days after billing_period_end 2026-03-10, more than 90"
    )


# --- total_positive ---------------------------------------------------------


def test_total_positive_passes_for_a_positive_total() -> None:
    assert check_total_positive(VALID) is None


@pytest.mark.parametrize("total", ["0.00", "-12.50"])
def test_total_positive_fails_for_zero_or_negative(total: str) -> None:
    assert check_total_positive(make_bill(total_amount=Decimal(total))) == (
        f"total_amount is {total}; expected a positive amount"
    )


# --- line_items_present -----------------------------------------------------


def test_line_items_present_passes_with_items() -> None:
    assert check_line_items_present(VALID) is None


def test_line_items_present_fails_when_empty() -> None:
    bill = make_bill(line_items=[])
    assert check_line_items_present(bill) == "line_items is empty, but total_amount is 152.20"
    assert validate_bill(bill).failed == ["line_items_sum", "line_items_present"]


# --- amount_precision -------------------------------------------------------


def test_amount_precision_passes_for_cents_and_trailing_zeros() -> None:
    assert check_amount_precision(make_bill(total_amount=Decimal("152.200"))) is None


def test_amount_precision_names_every_over_precise_amount() -> None:
    items = [*VALID.line_items[:-1], LineItem(description="State tax", amount=Decimal("5.475"))]
    bill = make_bill(line_items=items, total_amount=Decimal("152.195"))
    assert check_amount_precision(bill) == (
        "amounts with more than 2 decimal places: total_amount 152.195, line item 'State tax' 5.475"
    )


def test_amount_precision_ignores_usage_quantity() -> None:
    assert check_amount_precision(make_bill(usage_amount=Decimal("412.375"))) is None


# --- report -----------------------------------------------------------------


def test_report_lists_every_failure_in_check_order() -> None:
    bill = make_bill(
        billing_period_end=date(2026, 2, 1),
        due_date=date(2026, 1, 1),
        total_amount=Decimal("-1.001"),
    )

    report = validate_bill(bill)

    assert report.failed == ["line_items_sum", "billing_period_order", "due_date_window", "total_positive", "amount_precision"]
    assert report.passed == ["line_items_present"]
    assert not report.all_passed
    assert set(report.model_dump()) == {"passed", "failures", "failed", "all_passed"}


def test_a_check_that_errors_is_reported_as_failed_not_raised() -> None:
    def broken(bill: UtilityBill) -> str | None:
        raise ArithmeticError("boom")

    report = validate_bill(VALID, checks={"broken": broken, "total_positive": check_total_positive})

    assert report == ValidationReport(passed=["total_positive"], failures={"broken": "check could not run: ArithmeticError: boom"})
