# Evaluation results

Scope: first 5 test bills (smoke run), 5 bills. LLM model: `claude-haiku-4-5`. C / M / W = correct / missing / wrong. The test set is scored once; prompts are not tuned on it.

## Overall (all fields, all bills)

| Approach | Correct | Missing | Wrong | Bills fully correct |
|---|---|---|---|---|
| Regex baseline | 22 (44.0%) | 28 (56.0%) | 0 (0.0%) | 2/5 (40.0%) |
| LLM pipeline | 50 (100.0%) | 0 (0.0%) | 0 (0.0%) | 5/5 (100.0%) |

## Per template

The baseline parser has rules for the seen layouts only; the unseen layouts measure how it degrades.

| Template | Baseline layout | Bills | Regex fields C / M / W | Regex bills fully correct | LLM fields C / M / W | LLM bills fully correct |
|---|---|---|---|---|---|---|
| `bordered_table` | seen | 1 | 100.0% / 0.0% / 0.0% | 1/1 | 100.0% / 0.0% / 0.0% | 1/1 |
| `two_column` | seen | 1 | 100.0% / 0.0% / 0.0% | 1/1 | 100.0% / 0.0% / 0.0% | 1/1 |
| `two_page` | **unseen** | 2 | 0.0% / 100.0% / 0.0% | 0/2 | 100.0% / 0.0% / 0.0% | 2/2 |
| `receipt` | **unseen** | 1 | 20.0% / 80.0% / 0.0% | 0/1 | 100.0% / 0.0% / 0.0% | 1/1 |

## Per field (bill counts)

| Field | Regex baseline (C / M / W) | LLM pipeline (C / M / W) |
|---|---|---|
| `provider` | 2 / 3 / 0 | 5 / 0 / 0 |
| `account_number` | 2 / 3 / 0 | 5 / 0 / 0 |
| `billing_period_start` | 2 / 3 / 0 | 5 / 0 / 0 |
| `billing_period_end` | 2 / 3 / 0 | 5 / 0 / 0 |
| `due_date` | 2 / 3 / 0 | 5 / 0 / 0 |
| `usage_amount` | 3 / 2 / 0 | 5 / 0 / 0 |
| `usage_unit` | 3 / 2 / 0 | 5 / 0 / 0 |
| `line_items` | 2 / 3 / 0 | 5 / 0 / 0 |
| `total_amount` | 2 / 3 / 0 | 5 / 0 / 0 |
| `currency` | 2 / 3 / 0 | 5 / 0 / 0 |

## Line items, item level

Items are paired by normalised description (case-folded, surrounding whitespace stripped); see `match_line_items` in `src/evaluation/score.py`.

| Approach | Label items | Recovered | Right description, wrong amount | Missed | Invented |
|---|---|---|---|---|---|
| Regex baseline | 17 | 6 (35.3%) | 0 | 11 | 0 |
| LLM pipeline | 17 | 17 (100.0%) | 0 | 0 | 0 |

## Cost and latency

| Measure | Value |
|---|---|
| LLM model | claude-haiku-4-5 |
| Total API cost, all bills | $0.0142 |
| API cost of this run (new extractions only) | $0.0142 (5 bills) |
| LLM cost per bill (mean) | $0.00284 |
| LLM input tokens per bill (mean / median / max) | 2019 / 2018 / 2064 |
| LLM output tokens per bill (mean / median / max) | 164 / 161 / 193 |
| LLM latency per bill, s (mean / median / max) | 2.53 / 2.53 / 2.88 |
| Regex latency per bill, s (mean / median / max) | 0.007 / 0.007 / 0.009 |
| Regex cost per bill | $0 |

## Validation retries (LLM)

| Measure | Bills |
|---|---|
| Bills using 0 retries | 5 |
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
