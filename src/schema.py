"""Pydantic models describing the structured data extracted from a utility bill."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    """A single charge or credit listed on the bill."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    description: str
    """The label printed next to the charge, e.g. "Delivery charge" or "State tax"."""

    amount: Decimal
    """The money value of this line in the bill's currency. Negative for credits,
    discounts, and adjustments in the customer's favour."""


class UtilityBill(BaseModel):
    """One utility bill, as issued by a provider for a single billing period."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    provider: str
    """The company that issued the bill, e.g. "Pacific Gas and Electric". Usually
    shown in the header or logo."""

    account_number: str
    """The identifier the provider uses for the customer's account. Kept as a string
    because real account numbers contain leading zeros, dashes, and spaces."""

    billing_period_start: date
    """The first day of service this bill covers (often labelled "Service from")."""

    billing_period_end: date
    """The last day of service this bill covers (often labelled "Service to")."""

    due_date: date
    """The date payment must be received by to avoid late fees. Distinct from the
    statement/issue date."""

    usage_amount: Decimal
    """How much of the utility was consumed during the billing period, in the unit
    given by `usage_unit` — the metered quantity, not a money value."""

    usage_unit: str
    """The unit of measure for `usage_amount`, as printed on the bill, e.g. "kWh" for
    electricity, "therms" or "m3" for gas, "gal" or "CCF" for water."""

    line_items: list[LineItem]
    """The itemised charges and credits that make up the bill, in the order they
    appear. May not sum exactly to `total_amount` if the bill carries a previous
    balance or unitemised adjustments."""

    total_amount: Decimal
    """The total amount due for this bill (often labelled "Amount due" or "Total
    due"), in the currency given by `currency`."""

    currency: str = Field(pattern=r"^[A-Z]{3}$")
    """The ISO 4217 code for the currency all money values on the bill are in,
    e.g. "USD", "GBP", "EUR"."""
