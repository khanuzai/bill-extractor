"""Unit tests for the scoring rules. Hand-built bills, no API, no files."""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.evaluation.cost import cost_usd, prices_for
from src.evaluation.score import (
    FIELDS,
    ItemStats,
    Outcome,
    match_line_items,
    normalize_text,
    score_bill,
    score_field,
    values_match,
)
from src.parsers.regex_parser import ParseResult
from src.schema import LineItem, UtilityBill


def item(description: str, amount: str) -> LineItem:
    return LineItem(description=description, amount=Decimal(amount))


LABEL_ITEMS = [item("Energy charge", "95.12"), item("Autopay discount", "-10.00"), item("State tax", "5.48")]


def make_bill(**overrides: Any) -> UtilityBill:
    fields: dict[str, Any] = {
        "provider": "Lake Selena Power & Light",
        "account_number": "AC-31527-42",
        "billing_period_start": date(2026, 2, 8),
        "billing_period_end": date(2026, 3, 10),
        "due_date": None,
        "usage_amount": Decimal("412"),
        "usage_unit": "kWh",
        "line_items": LABEL_ITEMS,
        "total_amount": Decimal("90.60"),
        "currency": "USD",
    }
    return UtilityBill(**(fields | overrides))


LABEL = make_bill()


# --- strings ----------------------------------------------------------------


def test_normalize_text_only_folds_case_and_strips_the_ends() -> None:
    assert normalize_text("  Lake Selena POWER & Light\n") == "lake selena power & light"
    assert normalize_text("A  B") == "a  b"


@pytest.mark.parametrize("predicted", ["lake selena power & light", "  LAKE SELENA POWER & LIGHT  ", "\tLake Selena Power & Light\n"])
def test_strings_ignore_case_and_surrounding_whitespace(predicted: str) -> None:
    assert score_field(predicted, "Lake Selena Power & Light") is Outcome.CORRECT


@pytest.mark.parametrize(
    "predicted",
    ["Lake Selena Power and Light", "Lake Selena Power &  Light", "Lake Selena Power & Light Co", "LakeSelena Power & Light"],
)
def test_strings_with_different_words_or_inner_spacing_are_wrong(predicted: str) -> None:
    assert score_field(predicted, "Lake Selena Power & Light") is Outcome.WRONG


# --- dates, decimals, everything else ----------------------------------------


def test_dates_compare_exactly() -> None:
    assert score_field(date(2026, 3, 10), date(2026, 3, 10)) is Outcome.CORRECT
    assert score_field(date(2026, 3, 11), date(2026, 3, 10)) is Outcome.WRONG


def test_decimals_compare_exactly_by_value() -> None:
    assert score_field(Decimal("90.60"), Decimal("90.60")) is Outcome.CORRECT
    assert score_field(Decimal("90.6"), Decimal("90.60")) is Outcome.CORRECT
    assert score_field(Decimal("90.61"), Decimal("90.60")) is Outcome.WRONG
    assert score_field(Decimal("-90.60"), Decimal("90.60")) is Outcome.WRONG


def test_values_of_different_types_never_match() -> None:
    assert not values_match("90.60", Decimal("90.60"))
    assert not values_match("2026-03-10", date(2026, 3, 10))


# --- missing / None rules ----------------------------------------------------


def test_none_for_a_labelled_value_is_missing() -> None:
    assert score_field(None, "AC-31527-42") is Outcome.MISSING
    assert score_field(None, Decimal("90.60")) is Outcome.MISSING


def test_none_when_the_label_is_none_is_correct() -> None:
    assert score_field(None, None) is Outcome.CORRECT


def test_invented_value_when_the_label_is_none_is_wrong() -> None:
    assert score_field(date(2026, 3, 30), None) is Outcome.WRONG
    assert score_field("", None) is Outcome.WRONG


# --- line_items, all or nothing -----------------------------------------------


def test_line_items_identical_list_is_correct() -> None:
    assert score_field([item("energy charge ", "95.12"), *LABEL_ITEMS[1:]], LABEL_ITEMS) is Outcome.CORRECT


@pytest.mark.parametrize(
    "predicted",
    [
        LABEL_ITEMS[:2],  # one item dropped
        [*LABEL_ITEMS, item("Late fee", "5.00")],  # one item invented
        [LABEL_ITEMS[0], item("Autopay discount", "10.00"), LABEL_ITEMS[2]],  # sign error
        [LABEL_ITEMS[1], LABEL_ITEMS[0], LABEL_ITEMS[2]],  # reordered
        [],
    ],
    ids=["dropped", "invented", "sign error", "reordered", "empty"],
)
def test_line_items_any_difference_makes_the_whole_list_wrong(predicted: list[LineItem]) -> None:
    assert score_field(predicted, LABEL_ITEMS) is Outcome.WRONG


def test_line_items_none_is_missing() -> None:
    assert score_field(None, LABEL_ITEMS) is Outcome.MISSING


# --- item-level matching ------------------------------------------------------


def test_items_all_recovered() -> None:
    assert match_line_items(LABEL_ITEMS, LABEL_ITEMS) == ItemStats(
        label_items=3, predicted_items=3, recovered=3, wrong_amount=0, missed=0, invented=0
    )


def test_items_match_by_normalized_description_in_any_order() -> None:
    predicted = [item(" STATE TAX", "5.48"), item("energy charge", "95.12"), item("Autopay Discount  ", "-10.00")]
    assert match_line_items(predicted, LABEL_ITEMS).recovered == 3


def test_items_missed_invented_and_wrong_amount_are_counted_separately() -> None:
    predicted = [item("Energy charge", "95.12"), item("Autopay discount", "10.00"), item("Late fee", "4.00")]
    assert match_line_items(predicted, LABEL_ITEMS) == ItemStats(
        label_items=3, predicted_items=3, recovered=1, wrong_amount=1, missed=1, invented=1
    )


def test_items_with_a_different_description_do_not_match() -> None:
    stats = match_line_items([item("Energy charges", "95.12")], [item("Energy charge", "95.12")])
    assert (stats.recovered, stats.missed, stats.invented) == (0, 1, 1)


def test_repeated_descriptions_pair_exact_amounts_first() -> None:
    label = [item("Fee", "5.00"), item("Fee", "7.00")]
    stats = match_line_items([item("Fee", "7.00"), item("Fee", "5.00")], label)
    assert (stats.recovered, stats.wrong_amount) == (2, 0)

    stats = match_line_items([item("Fee", "7.00"), item("Fee", "9.00")], label)
    assert (stats.recovered, stats.wrong_amount, stats.missed, stats.invented) == (1, 1, 0, 0)


def test_items_none_counts_every_label_item_as_missed() -> None:
    assert match_line_items(None, LABEL_ITEMS) == ItemStats(
        label_items=3, predicted_items=0, recovered=0, wrong_amount=0, missed=3, invented=0
    )


# --- whole bills ----------------------------------------------------------------


def test_identical_bill_is_all_correct() -> None:
    score = score_bill(LABEL, LABEL)
    assert set(score.fields) == set(FIELDS)
    assert score.all_correct
    assert score.items.recovered == 3


def test_bill_with_a_wrong_field_and_an_invented_due_date() -> None:
    predicted = make_bill(total_amount=Decimal("100.60"), due_date=date(2026, 3, 30))

    score = score_bill(predicted, LABEL)

    assert score.fields["total_amount"] is Outcome.WRONG
    assert score.fields["due_date"] is Outcome.WRONG
    assert [name for name, outcome in score.fields.items() if outcome is Outcome.CORRECT] == [
        name for name in FIELDS if name not in ("total_amount", "due_date")
    ]
    assert not score.all_correct


def test_partial_parse_result_scores_found_fields_and_missing_rest() -> None:
    predicted = ParseResult(account_number="ac-31527-42", total_amount=Decimal("90.60"), currency="USD")

    score = score_bill(predicted, LABEL)

    assert score.fields["account_number"] is Outcome.CORRECT
    assert score.fields["total_amount"] is Outcome.CORRECT
    assert score.fields["due_date"] is Outcome.CORRECT  # None vs None
    assert score.fields["provider"] is Outcome.MISSING
    assert score.fields["line_items"] is Outcome.MISSING
    assert score.items.missed == 3


def test_failed_extraction_scores_none_labels_correct_and_the_rest_missing() -> None:
    score = score_bill(None, LABEL)
    assert score.fields["due_date"] is Outcome.CORRECT
    assert all(outcome is Outcome.MISSING for name, outcome in score.fields.items() if name != "due_date")


# --- cost -----------------------------------------------------------------------


def test_cost_uses_the_longest_matching_model_prefix() -> None:
    assert prices_for("claude-opus-5-5") == (Decimal("4"), Decimal("20"))
    assert prices_for("claude-opus-5") == (Decimal("5"), Decimal("25"))
    assert cost_usd("claude-haiku-4-5-20251001", 2_000_000, 100_000) == Decimal("2.5")
    assert cost_usd("some-other-model", 1, 1) is None
