"""Random, internally consistent utility bill data.

The structured `UtilityBill` is generated first and is the ground truth; PDFs are
rendered from it and nothing is ever parsed back out of them.

Amount rule
-----------
`total_amount` is exactly the sum of every line item's `amount`:

    bill.total_amount == sum(item.amount for item in bill.line_items)

Everything that changes what the customer owes is a line item, including a previous
balance (positive) and any credit or adjustment (negative). There are no hidden
charges. Every amount is a `Decimal` with exactly two decimal places, so the equality
is exact, not approximate. The total is always positive.

Each bill has 2-6 line items, in this order: previous balance (about 25% of bills),
1-3 charges, credit or adjustment (about 15%), tax. About 10% of bills have no due
date. Internet bills are flat fee: `usage_amount` and `usage_unit` are both None.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from faker import Faker

from src.schema import LineItem, UtilityBill

PREVIOUS_BALANCE_RATE = 0.25
CREDIT_RATE = 0.15
NO_DUE_DATE_RATE = 0.10

EARLIEST_PERIOD_START = date(2024, 1, 1)
LATEST_PERIOD_START = date(2026, 6, 30)

CENT = Decimal("0.01")


@dataclass(frozen=True)
class UtilityType:
    """What varies between kinds of utility when generating a bill."""

    name: str
    provider_patterns: tuple[str, ...]
    usage_unit: str | None
    usage_range: tuple[int, int]
    usage_decimals: int
    rate_range: tuple[float, float]
    primary_charges: tuple[str, ...]
    extra_charges: tuple[str, ...]


UTILITY_TYPES: tuple[UtilityType, ...] = (
    UtilityType(
        name="electricity",
        provider_patterns=("{city} Power & Light", "{surname} Electric Cooperative", "{state} Energy"),
        usage_unit="kWh",
        usage_range=(150, 1200),
        usage_decimals=0,
        rate_range=(0.11, 0.32),
        primary_charges=("Energy charge", "Electricity usage", "Electricity supply"),
        extra_charges=(
            "Delivery charge",
            "Basic customer charge",
            "Renewable energy rider",
            "Transmission charge",
        ),
    ),
    UtilityType(
        name="gas",
        provider_patterns=("{city} Natural Gas", "{surname} Gas Company", "{state} Gas & Heating"),
        usage_unit="m³",
        usage_range=(20, 250),
        usage_decimals=1,
        rate_range=(0.45, 1.40),
        primary_charges=("Gas supply charge", "Natural gas usage", "Gas consumption"),
        extra_charges=(
            "Distribution charge",
            "Fixed monthly charge",
            "Carbon levy",
            "Pipeline safety fee",
        ),
    ),
    UtilityType(
        name="water",
        provider_patterns=("{city} Water Authority", "{city} Municipal Water", "{surname} Valley Water District"),
        usage_unit="m³",
        usage_range=(8, 45),
        usage_decimals=1,
        rate_range=(1.80, 4.50),
        primary_charges=("Water consumption", "Water usage charge", "Metered water"),
        extra_charges=(
            "Sewer service",
            "Stormwater fee",
            "Meter service charge",
            "Infrastructure renewal fee",
        ),
    ),
    UtilityType(
        name="internet",
        provider_patterns=("{surname}Net Broadband", "{city} Fiber", "{state} Telecom"),
        usage_unit=None,
        usage_range=(0, 0),
        usage_decimals=0,
        rate_range=(0.0, 0.0),
        primary_charges=("Fiber 500 monthly plan", "Unlimited Internet 300", "Home Internet plan"),
        extra_charges=(
            "Modem rental",
            "Static IP address",
            "Regulatory recovery fee",
            "Wi-Fi extender rental",
        ),
    ),
)

# Currency code -> the label that currency's tax line carries.
TAX_DESCRIPTIONS: dict[str, tuple[str, ...]] = {
    "USD": ("State and local taxes", "Sales tax", "Utility users tax"),
    "CAD": ("GST/HST", "HST", "GST"),
}
CURRENCY_WEIGHTS: dict[str, int] = {"USD": 70, "CAD": 30}

PREVIOUS_BALANCE_DESCRIPTIONS: tuple[str, ...] = (
    "Previous balance",
    "Balance forward",
    "Unpaid balance from last bill",
)
CREDIT_DESCRIPTIONS: tuple[str, ...] = (
    "Loyalty credit",
    "Billing adjustment",
    "Autopay discount",
    "Service interruption credit",
)
ACCOUNT_NUMBER_PATTERNS: tuple[str, ...] = (
    "####-###-####",
    "0##########",
    "AC-#####-##",
    "### ### ####",
    "?#######",
)


@dataclass(frozen=True)
class BillExtras:
    """Details printed on a bill that are not part of the extraction schema.

    They make the PDFs realistic and give extractors plausible distractors (a second
    date, a second identifier) to get wrong.
    """

    customer_name: str
    address_lines: tuple[str, ...]
    statement_date: date
    meter_number: str | None


def make_faker(seed: int | None = None) -> Faker:
    """Return a Faker whose output is fully determined by `seed` (random if None)."""
    fake = Faker("en_US")
    if seed is not None:
        fake.seed_instance(seed)
    return fake


def _money(fake: Faker, low: float, high: float) -> Decimal:
    """A random amount between `low` and `high`, in whole cents."""
    cents = fake.random.randint(round(low * 100), round(high * 100))
    return Decimal(cents).scaleb(-2)


def _to_cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _usage(fake: Faker, utility: UtilityType) -> Decimal | None:
    if utility.usage_unit is None:
        return None
    low, high = utility.usage_range
    scale = 10**utility.usage_decimals
    return Decimal(fake.random.randint(low * scale, high * scale)) / scale


def _provider(fake: Faker, utility: UtilityType) -> str:
    pattern = fake.random.choice(utility.provider_patterns)
    return pattern.format(city=fake.city(), surname=fake.last_name(), state=fake.state())


def generate_bill(fake: Faker) -> UtilityBill:
    """Return a random valid bill obeying the amount rule in the module docstring."""
    rng = fake.random
    utility = rng.choice(UTILITY_TYPES)
    currency = rng.choices(list(CURRENCY_WEIGHTS), weights=list(CURRENCY_WEIGHTS.values()))[0]

    period_start = fake.date_between_dates(EARLIEST_PERIOD_START, LATEST_PERIOD_START)
    period_end = period_start + timedelta(days=rng.randint(27, 32))
    due_date: date | None = period_end + timedelta(days=rng.randint(14, 28))
    if rng.random() < NO_DUE_DATE_RATE:
        due_date = None

    usage_amount = _usage(fake, utility)
    if usage_amount is None:
        primary_amount = _money(fake, 39.99, 119.99)
    else:
        primary_amount = _to_cents(usage_amount * _money(fake, *utility.rate_range))

    charges = [LineItem(description=rng.choice(utility.primary_charges), amount=primary_amount)]
    for description in rng.sample(utility.extra_charges, k=rng.randint(0, 2)):
        charges.append(LineItem(description=description, amount=_money(fake, 2.50, 28.00)))
    charges_total = sum((item.amount for item in charges), Decimal(0))

    line_items: list[LineItem] = []
    if rng.random() < PREVIOUS_BALANCE_RATE:
        line_items.append(
            LineItem(
                description=rng.choice(PREVIOUS_BALANCE_DESCRIPTIONS),
                amount=_money(fake, 12.00, 240.00),
            )
        )
    line_items.extend(charges)
    if rng.random() < CREDIT_RATE:
        # Capped well below the charges so the total always stays positive.
        credit = _to_cents(charges_total * _money(fake, 0.05, 0.30))
        line_items.append(LineItem(description=rng.choice(CREDIT_DESCRIPTIONS), amount=-credit))
    tax = _to_cents(charges_total * _money(fake, 0.03, 0.13))
    line_items.append(LineItem(description=rng.choice(TAX_DESCRIPTIONS[currency]), amount=tax))

    return UtilityBill(
        provider=_provider(fake, utility),
        account_number=fake.bothify(rng.choice(ACCOUNT_NUMBER_PATTERNS), letters="ABCDEFGHJKLMNPRSTUVWXYZ"),
        billing_period_start=period_start,
        billing_period_end=period_end,
        due_date=due_date,
        usage_amount=usage_amount,
        usage_unit=utility.usage_unit,
        line_items=line_items,
        total_amount=sum((item.amount for item in line_items), Decimal(0)),
        currency=currency,
    )


def generate_extras(fake: Faker, bill: UtilityBill) -> BillExtras:
    """Return the off-schema details (customer, statement date, meter) printed on `bill`."""
    return BillExtras(
        customer_name=fake.name(),
        address_lines=tuple(fake.address().splitlines()),
        statement_date=bill.billing_period_end + timedelta(days=fake.random.randint(1, 6)),
        meter_number=fake.numerify("M-########") if bill.usage_amount is not None else None,
    )
