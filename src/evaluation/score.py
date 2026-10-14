"""Score one prediction against its ground-truth label. Pure functions, no I/O.

Field rules
-----------
- Strings are compared case-folded and stripped of surrounding whitespace. Nothing
  else is normalised: different words, or different spacing inside the string, are a
  mismatch.
- Dates, Decimals and everything else are compared for exact equality (Decimals by
  value, so 12.5 == 12.50).
- `line_items` is all-or-nothing: correct only if the predicted list has the same
  length as the label and every item matches the label item in the same position
  (description by the string rule, amount exactly).

Each field gets one outcome:

- label has a value, prediction equal  -> correct
- label has a value, prediction None   -> missing
- label has a value, prediction differs -> wrong
- label is None, prediction None       -> correct
- label is None, prediction has a value -> wrong

A failed extraction (no prediction at all) is scored as if every field were None, so
fields whose label is None still count as correct.
"""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, computed_field

from src.parsers.regex_parser import ParseResult
from src.schema import LineItem, UtilityBill

FIELDS: tuple[str, ...] = tuple(UtilityBill.model_fields)

Prediction = UtilityBill | ParseResult


class Outcome(StrEnum):
    CORRECT = "correct"
    MISSING = "missing"
    WRONG = "wrong"


class ItemStats(BaseModel):
    """Item-level line-item comparison for one bill (see `match_line_items`)."""

    label_items: int
    predicted_items: int
    recovered: int
    """Label items matched by a predicted item with the same description and amount."""
    wrong_amount: int
    """Label items matched by description, but with a different amount."""
    missed: int
    """Label items no predicted item matched by description."""
    invented: int
    """Predicted items that matched no label item by description."""


class BillScore(BaseModel):
    """The outcome for every field of one bill, plus item-level line-item stats."""

    fields: dict[str, Outcome]
    items: ItemStats

    @computed_field
    @property
    def all_correct(self) -> bool:
        return all(outcome is Outcome.CORRECT for outcome in self.fields.values())


def normalize_text(value: str) -> str:
    """Case-fold and strip surrounding whitespace; nothing else."""
    return value.strip().casefold()


def _item_key(item: LineItem) -> tuple[str, Decimal]:
    return normalize_text(item.description), item.amount


def values_match(predicted: object, expected: object) -> bool:
    """Whether a predicted field value matches the label value under the field rules."""
    if isinstance(predicted, str) and isinstance(expected, str):
        return normalize_text(predicted) == normalize_text(expected)
    if isinstance(predicted, list) and isinstance(expected, list):
        return len(predicted) == len(expected) and all(
            _item_key(p) == _item_key(e) for p, e in zip(predicted, expected, strict=True)
        )
    return predicted == expected


def score_field(predicted: object, expected: object) -> Outcome:
    """Apply the correct / missing / wrong rules to one field."""
    if expected is None:
        return Outcome.CORRECT if predicted is None else Outcome.WRONG
    if predicted is None:
        return Outcome.MISSING
    return Outcome.CORRECT if values_match(predicted, expected) else Outcome.WRONG


def match_line_items(predicted: list[LineItem] | None, expected: list[LineItem]) -> ItemStats:
    """Pair predicted line items with label items, one-to-one, ignoring order.

    Items are matched by normalised description (case-folded, surrounding whitespace
    stripped). Pairing runs in two passes so repeated descriptions pair sensibly:
    first each label item takes an unpaired predicted item with the same description
    *and* amount; then each still-unpaired label item takes any unpaired predicted item
    with the same description. Within a pass, items are taken in list order.

    A label item paired in the first pass is `recovered`; in the second pass,
    `wrong_amount`; never paired, `missed`. A predicted item never paired is `invented`.
    None (the approach found no line items) counts as an empty list.
    """
    remaining = list(predicted or [])
    unpaired_labels: list[LineItem] = []
    recovered = 0
    for label_item in expected:
        key = _item_key(label_item)
        index = next((i for i, item in enumerate(remaining) if _item_key(item) == key), None)
        if index is None:
            unpaired_labels.append(label_item)
        else:
            remaining.pop(index)
            recovered += 1

    wrong_amount = 0
    for label_item in unpaired_labels:
        description = normalize_text(label_item.description)
        index = next(
            (i for i, item in enumerate(remaining) if normalize_text(item.description) == description), None
        )
        if index is not None:
            remaining.pop(index)
            wrong_amount += 1

    return ItemStats(
        label_items=len(expected),
        predicted_items=len(predicted or []),
        recovered=recovered,
        wrong_amount=wrong_amount,
        missed=len(unpaired_labels) - wrong_amount,
        invented=len(remaining),
    )


def score_bill(prediction: Prediction | None, label: UtilityBill) -> BillScore:
    """Score every field of `prediction` against `label`; None means extraction failed."""
    fields = {
        name: score_field(getattr(prediction, name) if prediction is not None else None, getattr(label, name))
        for name in FIELDS
    }
    items = match_line_items(prediction.line_items if prediction is not None else None, label.line_items)
    return BillScore(fields=fields, items=items)
