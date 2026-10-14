# Evaluation results

Scope: full test set, 100 bills. LLM model: `claude-haiku-4-5`. C / M / W = correct / missing / wrong. The test set is scored once; prompts are not tuned on it.

## Overall (all fields, all bills)

| Approach | Correct | Missing | Wrong | Bills fully correct |
|---|---|---|---|---|
| Regex baseline | 435 (43.5%) | 565 (56.5%) | 0 (0.0%) | 40/100 (40.0%) |
| LLM pipeline | 1000 (100.0%) | 0 (0.0%) | 0 (0.0%) | 100/100 (100.0%) |

## Per template

The baseline parser has rules for the seen layouts only; the unseen layouts measure how it degrades.

| Template | Baseline layout | Bills | Regex fields C / M / W | Regex bills fully correct | LLM fields C / M / W | LLM bills fully correct |
|---|---|---|---|---|---|---|
| `bordered_table` | seen | 20 | 100.0% / 0.0% / 0.0% | 20/20 | 100.0% / 0.0% / 0.0% | 20/20 |
| `two_column` | seen | 20 | 100.0% / 0.0% / 0.0% | 20/20 | 100.0% / 0.0% / 0.0% | 20/20 |
| `summary_box` | **unseen** | 20 | 4.5% / 95.5% / 0.0% | 0/20 | 100.0% / 0.0% / 0.0% | 20/20 |
| `two_page` | **unseen** | 20 | 8.5% / 91.5% / 0.0% | 0/20 | 100.0% / 0.0% / 0.0% | 20/20 |
| `receipt` | **unseen** | 20 | 4.5% / 95.5% / 0.0% | 0/20 | 100.0% / 0.0% / 0.0% | 20/20 |

## Per field (bill counts)

| Field | Regex baseline (C / M / W) | LLM pipeline (C / M / W) |
|---|---|---|
| `provider` | 40 / 60 / 0 | 100 / 0 / 0 |
| `account_number` | 40 / 60 / 0 | 100 / 0 / 0 |
| `billing_period_start` | 40 / 60 / 0 | 100 / 0 / 0 |
| `billing_period_end` | 40 / 60 / 0 | 100 / 0 / 0 |
| `due_date` | 45 / 55 / 0 | 100 / 0 / 0 |
| `usage_amount` | 55 / 45 / 0 | 100 / 0 / 0 |
| `usage_unit` | 55 / 45 / 0 | 100 / 0 / 0 |
| `line_items` | 40 / 60 / 0 | 100 / 0 / 0 |
| `total_amount` | 40 / 60 / 0 | 100 / 0 / 0 |
| `currency` | 40 / 60 / 0 | 100 / 0 / 0 |

## Line items, item level

Items are paired by normalised description (case-folded, surrounding whitespace stripped); see `match_line_items` in `src/evaluation/score.py`.

| Approach | Label items | Recovered | Right description, wrong amount | Missed | Invented |
|---|---|---|---|---|---|
| Regex baseline | 345 | 135 (39.1%) | 0 | 210 | 0 |
| LLM pipeline | 345 | 345 (100.0%) | 0 | 0 | 0 |

## Cost and latency

| Measure | Value |
|---|---|
| LLM model | claude-haiku-4-5 |
| Total API cost, all bills | $0.2802 |
| API cost of this run (new extractions only) | $0.2660 (95 bills) |
| LLM cost per bill (mean) | $0.00280 |
| LLM input tokens per bill (mean / median / max) | 2011 / 2015 / 2068 |
| LLM output tokens per bill (mean / median / max) | 158 / 158 / 211 |
| LLM latency per bill, s (mean / median / max) | 2.45 / 2.44 / 3.09 |
| Regex latency per bill, s (mean / median / max) | 0.007 / 0.007 / 0.013 |
| Regex cost per bill | $0 |

## Validation retries (LLM)

| Measure | Bills |
|---|---|
| Bills using 0 retries | 100 |
| Bills using 1 retries | 0 |
| Bills using 2 retries | 0 |
| Bills fixed by a retry (all checks pass at the end) | 0 |
| Bills still failing a check after retries (kept, best attempt) | 0 |
| Bills with no valid extraction | 0 |

## Validation checks fired (LLM, bill counts)

| Check | Failed on first attempt | Failed on any attempt | Still failing at the end |
|---|---|---|---|
| `line_items_sum` | 0 | 0 | 0 |
| `billing_period_order` | 0 | 0 | 0 |
| `due_date_window` | 0 | 0 | 0 |
| `total_positive` | 0 | 0 | 0 |
| `line_items_present` | 0 | 0 | 0 |
| `amount_precision` | 0 | 0 | 0 |
