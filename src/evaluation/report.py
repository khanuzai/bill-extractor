"""Render an evaluation run as markdown tables. Pure functions, no I/O."""

import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from decimal import Decimal

from src.evaluation.records import BillRecord, RunResults
from src.evaluation.score import FIELDS, BillScore, ItemStats, Outcome
from src.extraction.extract import MAX_RETRIES
from src.extraction.validate import CHECKS
from src.generator.render import TEMPLATES
from src.parsers.regex_parser import TARGET_TEMPLATES

APPROACHES: tuple[str, ...] = ("regex", "llm")
APPROACH_NAMES: dict[str, str] = {"regex": "Regex baseline", "llm": "LLM pipeline"}


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _score(bill: BillRecord, approach: str) -> BillScore:
    return bill.regex.score if approach == "regex" else bill.llm.score


def _outcomes(bills: Sequence[BillRecord], approach: str, field: str | None = None) -> Counter[Outcome]:
    counts: Counter[Outcome] = Counter()
    for bill in bills:
        fields = _score(bill, approach).fields
        counts.update([fields[field]] if field else fields.values())
    return counts


def _cmw(counts: Counter[Outcome]) -> str:
    """'correct% / missing% / wrong%' for a set of outcomes."""
    total = sum(counts.values())
    return " / ".join(_pct(counts[outcome], total) for outcome in Outcome)


def overall_table(bills: Sequence[BillRecord]) -> str:
    rows = []
    for approach in APPROACHES:
        counts = _outcomes(bills, approach)
        total = sum(counts.values())
        fully_correct = sum(_score(bill, approach).all_correct for bill in bills)
        rows.append(
            [
                APPROACH_NAMES[approach],
                *(f"{counts[outcome]} ({_pct(counts[outcome], total)})" for outcome in Outcome),
                f"{fully_correct}/{len(bills)} ({_pct(fully_correct, len(bills))})",
            ]
        )
    return _table(["Approach", "Correct", "Missing", "Wrong", "Bills fully correct"], rows)


def field_table(bills: Sequence[BillRecord]) -> str:
    rows = []
    for field in FIELDS:
        cells = []
        for approach in APPROACHES:
            counts = _outcomes(bills, approach, field)
            cells.append(" / ".join(str(counts[outcome]) for outcome in Outcome))
        rows.append([f"`{field}`", *cells])
    header = ["Field", *(f"{APPROACH_NAMES[a]} (C / M / W)" for a in APPROACHES)]
    return _table(header, rows)


def template_table(bills: Sequence[BillRecord]) -> str:
    rows = []
    for template in TEMPLATES:
        subset = [bill for bill in bills if bill.template == template]
        if not subset:
            continue
        layout = "seen" if template in TARGET_TEMPLATES else "**unseen**"
        cells: list[str] = []
        for approach in APPROACHES:
            fully_correct = sum(_score(bill, approach).all_correct for bill in subset)
            cells += [_cmw(_outcomes(subset, approach)), f"{fully_correct}/{len(subset)}"]
        rows.append([f"`{template}`", layout, len(subset), *cells])
    header = [
        "Template",
        "Baseline layout",
        "Bills",
        "Regex fields C / M / W",
        "Regex bills fully correct",
        "LLM fields C / M / W",
        "LLM bills fully correct",
    ]
    return _table(header, rows)


def _summary(values: Sequence[float], decimals: int) -> str:
    """'mean / median / max' to the given number of decimals."""
    if not values:
        return "n/a"
    return " / ".join(f"{value:.{decimals}f}" for value in (statistics.mean(values), statistics.median(values), max(values)))


def cost_table(results: RunResults) -> str:
    bills = results.bills
    priced = [bill.llm.cost_usd for bill in bills if bill.llm.cost_usd is not None]
    mean_cost = f"${sum(priced, Decimal(0)) / len(priced):.5f}" if priced else "n/a"
    rows = [
        ["LLM model", results.model],
        ["Total API cost, all bills", f"${results.cost_total_usd:.4f}"],
        ["API cost of this run (new extractions only)", f"${results.cost_this_run_usd:.4f} ({results.new_api_calls} bills)"],
        ["LLM cost per bill (mean)", mean_cost],
        ["LLM input tokens per bill (mean / median / max)", _summary([b.llm.input_tokens for b in bills], 0)],
        ["LLM output tokens per bill (mean / median / max)", _summary([b.llm.output_tokens for b in bills], 0)],
        ["LLM latency per bill, s (mean / median / max)", _summary([b.llm.latency_seconds for b in bills], 2)],
        ["Regex latency per bill, s (mean / median / max)", _summary([b.regex.latency_seconds for b in bills], 3)],
        ["Regex cost per bill", "$0"],
    ]
    if results.unpriced_bills:
        rows.append(["Bills without a cost (model not in price table)", results.unpriced_bills])
    return _table(["Measure", "Value"], rows)


def _item_totals(bills: Sequence[BillRecord], approach: str) -> ItemStats:
    stats = [_score(bill, approach).items for bill in bills]
    return ItemStats(**{name: sum(getattr(s, name) for s in stats) for name in ItemStats.model_fields})


def item_table(bills: Sequence[BillRecord]) -> str:
    rows = []
    for approach in APPROACHES:
        totals = _item_totals(bills, approach)
        rows.append(
            [
                APPROACH_NAMES[approach],
                totals.label_items,
                f"{totals.recovered} ({_pct(totals.recovered, totals.label_items)})",
                totals.wrong_amount,
                totals.missed,
                totals.invented,
            ]
        )
    header = ["Approach", "Label items", "Recovered", "Right description, wrong amount", "Missed", "Invented"]
    return _table(header, rows)


def retry_table(bills: Sequence[BillRecord]) -> str:
    retries = Counter(bill.llm.retries for bill in bills)
    rows: list[list[object]] = [[f"Bills using {n} retries", retries[n]] for n in range(MAX_RETRIES + 1)]
    fixed = sum(
        1
        for bill in bills
        if bill.llm.retries > 0 and bill.llm.validation is not None and bill.llm.validation.all_passed
    )
    still_failing = sum(1 for bill in bills if bill.llm.validation is not None and not bill.llm.validation.all_passed)
    rows += [
        ["Bills fixed by a retry (all checks pass at the end)", fixed],
        ["Bills still failing a check after retries (kept, best attempt)", still_failing],
        ["Bills with no valid extraction", sum(1 for bill in bills if bill.llm.validation is None)],
    ]
    return _table(["Measure", "Bills"], rows)


def check_table(bills: Sequence[BillRecord]) -> str:
    first = Counter[str]()
    any_attempt = Counter[str]()
    final = Counter[str]()
    for bill in bills:
        attempts = bill.llm.attempts
        if attempts:
            first.update(attempts[0].failed_checks)
        any_attempt.update({check for attempt in attempts for check in attempt.failed_checks})
        if bill.llm.validation is not None:
            final.update(bill.llm.validation.failed)
    rows = [[f"`{check}`", first[check], any_attempt[check], final[check]] for check in CHECKS]
    return _table(["Check", "Failed on first attempt", "Failed on any attempt", "Still failing at the end"], rows)


def render_report(results: RunResults) -> str:
    """The full markdown report for one run."""
    bills = results.bills
    scope = f"first {results.limit} test bills (smoke run)" if results.limit else "full test set"
    sections = [
        "# Evaluation results",
        f"Scope: {scope}, {len(bills)} bills. LLM model: `{results.model}`. "
        "C / M / W = correct / missing / wrong. The test set is scored once; prompts are not tuned on it.",
        "## Overall (all fields, all bills)",
        overall_table(bills),
        "## Per template",
        "The baseline parser has rules for the seen layouts only; the unseen layouts measure how it degrades.",
        template_table(bills),
        "## Per field (bill counts)",
        field_table(bills),
        "## Line items, item level",
        "Items are paired by normalised description (case-folded, surrounding whitespace stripped); "
        "see `match_line_items` in `src/evaluation/score.py`.",
        item_table(bills),
        "## Cost and latency",
        cost_table(results),
        "## Validation retries (LLM)",
        retry_table(bills),
        "## Validation checks fired (LLM, bill counts)",
        check_table(bills),
    ]
    return "\n\n".join(sections) + "\n"
