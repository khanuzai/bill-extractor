# Baseline regex parser

The hand-written "before" baseline that the LLM extraction pipeline is compared
against. It stands in for the per-provider parsers you would write by hand in
production: one set of label-anchored regex rules per known bill layout.

## Targeted templates

Rules exist for exactly two of the five generator templates:

| Template | Anchors the rules rely on |
|---|---|
| `bordered_table` | `Account Number`, `Service Period`, `Usage This Period`, `Due Date`, `Description … Amount (CUR)`, `Amount Due`, `Contact … customer care` |
| `two_column` | `Account`, `Billing period from` / `to`, `Consumption`, `Payment due`, `What you owe`, `Total Balance`, `Thank you for choosing …` |

Date formats handled: `03/15/2026` and `March 15, 2026` (the two these templates
print). Money formats handled: plain amounts plus all three credit notations the
generator produces (`-7.33`, `(87.74)`, `26.44 CR`), with optional `$` and thousands
separators.

## Deliberately excluded templates

`summary_box`, `two_page` and `receipt` have **no rules, on purpose**. They are the
unseen layouts: the evaluation uses them to measure how a hand-written parser degrades
on formats it was not built for, which is the case the LLM pipeline is meant to win.

Do not add rules, anchors, date formats or special cases aimed at those three
templates, and do not tune the existing rules against their output. Doing so would
turn the baseline into a parser that has seen the whole test set and make the
comparison meaningless. If a new layout needs baseline coverage, add a new template to
the generator and decide explicitly which side of the seen/unseen line it belongs on.

## Behaviour

- `text.extract_text(pdf_path)` returns the text of all pages in reading order (top to
  bottom, left to right), so a label and its value share a line.
- `regex_parser.parse_bill_text(text)` returns a `ParseResult`: every `UtilityBill`
  field, all optional, plus `found`, the list of fields that were located.
- A field is `None` unless its label was found and its value parsed cleanly. The
  parser never guesses and never raises on an unrecognised layout; an unknown bill
  comes back with `found == []`.
- Line items are all or nothing: if any row between the charges heading and the total
  is not `description  amount`, `line_items` is `None`.
