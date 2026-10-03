"""Hand-written regex baseline parser for bill text.

Every rule is anchored on a label printed by one of the two layouts this parser was
written for (`bordered_table` and `two_column`; see README.md in this package). A
field whose label is not found, or whose value does not parse cleanly, is None: the
parser never guesses, and never raises on a layout it does not recognise.
"""

import re
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, computed_field

from src.schema import LineItem

# The gap between a label and the value printed beside it on the same line.
GAP = r"[ \t]{2,}"

US_DATE = r"\d{2}/\d{2}/\d{4}"
LONG_DATE = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
MONEY = r"-?\$?\(?\d[\d,]*\.\d{2}\)?(?: ?CR)?"
CURRENCY = r"[A-Z]{3}"
QUANTITY = r"\d[\d,]*(?:\.\d+)?"

MONTHS: dict[str, int] = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}

_US_DATE_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
_LONG_DATE_RE = re.compile(r"([A-Z][a-z]+) (\d{1,2}), (\d{4})")
_MONEY_RE = re.compile(
    r"(?P<minus>-)?\$?(?P<paren>\()?(?P<number>\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2})(?(paren)\))(?P<cr> ?CR)?"
)
_ACCOUNT_NUMBER_RE = re.compile(r"(?=.*\d)[A-Z0-9][A-Z0-9 -]{3,}[A-Z0-9]")


class ParseResult(BaseModel):
    """What the regex parser could locate in a bill. Any field may be None."""

    provider: str | None = None
    account_number: str | None = None
    billing_period_start: date | None = None
    billing_period_end: date | None = None
    due_date: date | None = None
    usage_amount: Decimal | None = None
    usage_unit: str | None = None
    line_items: list[LineItem] | None = None
    total_amount: Decimal | None = None
    currency: str | None = None

    @computed_field
    @property
    def found(self) -> list[str]:
        """Names of the fields the parser located, in schema order."""
        return [name for name in type(self).model_fields if getattr(self, name) is not None]


def parse_money(text: str) -> Decimal | None:
    """Convert a printed amount to a signed Decimal, or None if it is not one.

    Credits may be written with a leading minus (-7.33), in accounting parentheses
    ((87.74)) or with a CR suffix (26.44 CR); all three come back negative.
    """
    match = _MONEY_RE.fullmatch(text.strip())
    if match is None:
        return None
    amount = Decimal(match["number"].replace(",", ""))
    is_credit = bool(match["minus"] or match["paren"] or match["cr"])
    return -amount if is_credit else amount


def parse_date(text: str) -> date | None:
    """Parse 03/15/2026 or March 15, 2026; None for anything else."""
    text = text.strip()
    if match := _US_DATE_RE.fullmatch(text):
        month, day, year = int(match[1]), int(match[2]), int(match[3])
    elif (match := _LONG_DATE_RE.fullmatch(text)) and match[1] in MONTHS:
        month, day, year = MONTHS[match[1]], int(match[2]), int(match[3])
    else:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_quantity(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _search(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    """Return the first match of the first pattern that matches at all."""
    for pattern in patterns:
        if match := re.search(pattern, text, flags=re.MULTILINE):
            return match
    return None


def _provider(text: str) -> str | None:
    flowed = " ".join(text.split())
    match = _search(
        (
            r"Contact (.+?) customer care and quote your account",
            r"Thank you for choosing (.+)\.$",
        ),
        flowed,
    )
    return match[1] if match else None


def _account_number(text: str) -> str | None:
    match = _search(
        (
            # The number sits alone on the line under its label.
            r"Account Number[ \t]*\n[ \t]*(\S[^\n]*?)[ \t]*$",
            rf"^[ \t]*Account{GAP}(\S.*?)[ \t]*$",
        ),
        text,
    )
    if match is None or _ACCOUNT_NUMBER_RE.fullmatch(match[1]) is None:
        return None
    return match[1]


def _billing_period(text: str) -> tuple[date | None, date | None]:
    if match := _search((rf"^[ \t]*Service Period{GAP}({US_DATE}) - ({US_DATE})[ \t]*$",), text):
        start, end = parse_date(match[1]), parse_date(match[2])
    else:
        start_match = _search((rf"^[ \t]*Billing period from{GAP}({LONG_DATE})[ \t]*$",), text)
        end_match = _search((rf"^[ \t]*Billing period to{GAP}({LONG_DATE})[ \t]*$",), text)
        start = parse_date(start_match[1]) if start_match else None
        end = parse_date(end_match[1]) if end_match else None
    if start is None or end is None or start > end:
        return None, None
    return start, end


def _due_date(text: str) -> date | None:
    match = _search(
        (
            rf"^[ \t]*Due Date{GAP}({US_DATE})[ \t]*$",
            rf"^[ \t]*Payment due{GAP}({LONG_DATE})[ \t]*$",
        ),
        text,
    )
    return parse_date(match[1]) if match else None


def _usage(text: str) -> tuple[Decimal | None, str | None]:
    match = _search(
        (
            rf"^[ \t]*Usage This Period{GAP}({QUANTITY}) (\S+)[ \t]*$",
            rf"^[ \t]*Consumption{GAP}({QUANTITY}) (\S+)[ \t]*$",
        ),
        text,
    )
    if match is None:
        return None, None
    return _parse_quantity(match[1]), match[2]


def _total_amount(text: str) -> Decimal | None:
    match = _search(
        (
            rf"^[ \t]*Amount Due{GAP}({MONEY})[ \t]*$",
            rf"^[ \t]*Total Balance{GAP}({MONEY}) {CURRENCY}[ \t]*$",
        ),
        text,
    )
    return parse_money(match[1]) if match else None


def _currency(text: str) -> str | None:
    match = _search(
        (
            rf"^[ \t]*Description{GAP}Amount \(({CURRENCY})\)[ \t]*$",
            rf"^[ \t]*Total Balance{GAP}{MONEY} ({CURRENCY})[ \t]*$",
        ),
        text,
    )
    return match[1] if match else None


def _line_items(text: str) -> list[LineItem] | None:
    """Parse the rows between a charges heading and the total line.

    All or nothing: if any row in the block is not `description  amount`, the block is
    not the table this rule was written for, so nothing is returned.
    """
    match = re.search(
        rf"^[ \t]*Description{GAP}Amount \({CURRENCY}\)[ \t]*\n(.*?)^[ \t]*Amount Due{GAP}"
        rf"|^[ \t]*What you owe[ \t]*\n(.*?)^[ \t]*Total Balance{GAP}",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return None
    block = match[1] if match[1] is not None else match[2]
    items: list[LineItem] = []
    for line in block.splitlines():
        if not line.strip():
            continue
        row = re.fullmatch(rf"[ \t]*(\S.*?){GAP}({MONEY})[ \t]*", line)
        amount = parse_money(row[2]) if row else None
        if row is None or amount is None:
            return None
        items.append(LineItem(description=row[1], amount=amount))
    return items or None


def parse_bill_text(text: str) -> ParseResult:
    """Parse the text of a bill PDF into whichever fields can be located with confidence."""
    period_start, period_end = _billing_period(text)
    usage_amount, usage_unit = _usage(text)
    return ParseResult(
        provider=_provider(text),
        account_number=_account_number(text),
        billing_period_start=period_start,
        billing_period_end=period_end,
        due_date=_due_date(text),
        usage_amount=usage_amount,
        usage_unit=usage_unit,
        line_items=_line_items(text),
        total_amount=_total_amount(text),
        currency=_currency(text),
    )
