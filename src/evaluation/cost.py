"""API cost of an extraction, from token counts. Pure functions."""

from decimal import Decimal

# USD per million tokens (input, output), Anthropic first-party list prices as of
# 2026-09. Keys are model-id prefixes: responses report dated ids such as
# "claude-haiku-4-5-20251001".
PRICES_PER_MILLION_TOKENS: dict[str, tuple[Decimal, Decimal]] = {
    "claude-fable-5-1": (Decimal("10"), Decimal("50")),
    "claude-opus-5-5": (Decimal("4"), Decimal("20")),
    "claude-opus-5": (Decimal("5"), Decimal("25")),
    "claude-opus-4-8": (Decimal("5"), Decimal("25")),
    "claude-opus-4-7": (Decimal("5"), Decimal("25")),
    "claude-opus-4-6": (Decimal("5"), Decimal("25")),
    "claude-sonnet-5-5": (Decimal("2"), Decimal("10")),
    "claude-sonnet-5": (Decimal("2"), Decimal("10")),
    "claude-sonnet-4-6": (Decimal("3"), Decimal("15")),
    "claude-haiku-4-5": (Decimal("1"), Decimal("5")),
}

MILLION = Decimal(1_000_000)


def prices_for(model: str) -> tuple[Decimal, Decimal] | None:
    """The (input, output) price per million tokens for `model`, or None if unknown.

    The longest matching prefix wins, so "claude-opus-5-5" is not priced as
    "claude-opus-5".
    """
    for prefix in sorted(PRICES_PER_MILLION_TOKENS, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES_PER_MILLION_TOKENS[prefix]
    return None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    """The cost of the given usage in USD, or None if the model's prices are unknown."""
    prices = prices_for(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / MILLION
