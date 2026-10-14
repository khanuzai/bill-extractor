"""Consistency checks on an extracted bill that need no ground-truth label.

Each check takes a `UtilityBill` and returns None if it passes, or a failure message
naming the values involved. `validate_bill` runs them all and never raises.
"""

from collections.abc import Callable
from decimal import Decimal

from pydantic import BaseModel, computed_field

from src.schema import UtilityBill

MAX_DAYS_DUE_AFTER_PERIOD_END = 90
CENT = Decimal("0.01")

Check = Callable[[UtilityBill], str | None]


def check_line_items_sum(bill: UtilityBill) -> str | None:
    """The line items add up exactly to the total."""
    line_sum = sum((item.amount for item in bill.line_items), Decimal(0))
    if line_sum == bill.total_amount:
        return None
    return (
        f"line items sum to {line_sum} but total_amount is {bill.total_amount} "
        f"(difference {bill.total_amount - line_sum})"
    )


def check_billing_period_order(bill: UtilityBill) -> str | None:
    """The billing period ends after it starts."""
    if bill.billing_period_end > bill.billing_period_start:
        return None
    return (
        f"billing_period_end {bill.billing_period_end} is not after "
        f"billing_period_start {bill.billing_period_start}"
    )


def check_due_date_window(bill: UtilityBill) -> str | None:
    """A due date, if present, falls 0 to 90 days after the billing period ends."""
    if bill.due_date is None:
        return None
    days_after = (bill.due_date - bill.billing_period_end).days
    if days_after < 0:
        return (
            f"due_date {bill.due_date} is {-days_after} days before "
            f"billing_period_end {bill.billing_period_end}"
        )
    if days_after > MAX_DAYS_DUE_AFTER_PERIOD_END:
        return (
            f"due_date {bill.due_date} is {days_after} days after billing_period_end "
            f"{bill.billing_period_end}, more than {MAX_DAYS_DUE_AFTER_PERIOD_END}"
        )
    return None


def check_total_positive(bill: UtilityBill) -> str | None:
    """The total amount due is greater than zero."""
    if bill.total_amount > 0:
        return None
    return f"total_amount is {bill.total_amount}; expected a positive amount"


def check_line_items_present(bill: UtilityBill) -> str | None:
    """The bill has at least one line item."""
    if bill.line_items:
        return None
    return f"line_items is empty, but total_amount is {bill.total_amount}"


def check_amount_precision(bill: UtilityBill) -> str | None:
    """Every money amount (total and line items) has at most 2 decimal places."""
    amounts = [("total_amount", bill.total_amount)] + [
        (f"line item {item.description!r}", item.amount) for item in bill.line_items
    ]
    too_precise = [f"{label} {amount}" for label, amount in amounts if amount != amount.quantize(CENT)]
    if not too_precise:
        return None
    return "amounts with more than 2 decimal places: " + ", ".join(too_precise)


CHECKS: dict[str, Check] = {
    "line_items_sum": check_line_items_sum,
    "billing_period_order": check_billing_period_order,
    "due_date_window": check_due_date_window,
    "total_positive": check_total_positive,
    "line_items_present": check_line_items_present,
    "amount_precision": check_amount_precision,
}


class ValidationReport(BaseModel):
    """The outcome of running every check on one bill."""

    passed: list[str]
    """Names of the checks that passed, in CHECKS order."""

    failures: dict[str, str]
    """Failed check name -> failure message, in CHECKS order."""

    @computed_field
    @property
    def failed(self) -> list[str]:
        """Names of the checks that failed, in CHECKS order."""
        return list(self.failures)

    @computed_field
    @property
    def all_passed(self) -> bool:
        return not self.failures


def validate_bill(bill: UtilityBill, checks: dict[str, Check] = CHECKS) -> ValidationReport:
    """Run every check on `bill`. Never raises: a check that errors counts as failed."""
    passed: list[str] = []
    failures: dict[str, str] = {}
    for name, check in checks.items():
        try:
            message = check(bill)
        except (ArithmeticError, TypeError, ValueError) as exc:
            message = f"check could not run: {type(exc).__name__}: {exc}"
        if message is None:
            passed.append(name)
        else:
            failures[name] = message
    return ValidationReport(passed=passed, failures=failures)
