"""Evaluate the RAG pipeline on 30 questions over the ingested collection.

    python -m src.storage.ingest --split dev && python -m src.rag.eval --split dev
    python -m src.storage.ingest --split test && python -m src.rag.eval --split test

The 30 questions are fixed templates. Each is filled in from the labels of the
ingested bills by a deterministic rule ("the second bill with a credit"), so the same
questions run on the dev and test collections and every expected answer comes from
the ground-truth labels, never from a hand-typed number.

Measures
--------
- Routing accuracy: the router's decision (rules, then the model) against each
  question's labelled route, over all 30.
- Recall@5 (answerable semantic questions): a hit if any of the top 5 chunks comes
  from the labelled bill and contains the question's evidence text (compared
  case-folded, with whitespace collapsed). Bill recall@5, the looser "a chunk from
  the right bill is in the top 5", is reported alongside. Retrieval is limited to a
  bill the question names (see `retrieve`), so this mostly measures finding the
  right section of the right bill.
- Exact match (aggregate questions): the SQL result's first row contains the
  expected value. Numbers are compared rounded to 2 decimal places; text
  case-folded. A "which bill" question accepts the bill's id, account number or
  provider; ties accept any tied bill.
- Abstention (unanswerable questions): the answer says the bills don't contain it.

Retrieval and SQL are measured on the labelled route (passed to the pipeline), so a
routing mistake is counted once, in routing accuracy, rather than again downstream.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import anthropic
import psycopg
from pydantic import BaseModel

from src.evaluation.cost import cost_usd
from src.extraction.client import ExtractionConfig, load_config, make_client
from src.generator.data import UTILITY_TYPES
from src.rag.answer import Answer, answer_question
from src.rag.embed import Embedder, load_embedder
from src.rag.retrieve import DEFAULT_TOP_K
from src.rag.router import Route, route_question
from src.schema import BillLabel, LineItem
from src.storage.db import connect, load_settings, stored_bill_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = REPO_ROOT / "data" / "labels"
RESULTS_DIR = REPO_ROOT / "data" / "results"
RESULT_STEMS: dict[str, str] = {"dev": "rag_results_dev", "test": "rag_results"}

CENT = Decimal("0.01")
EXTRA_CHARGES = frozenset(d for utility in UTILITY_TYPES for d in utility.extra_charges)


# --- question instances -------------------------------------------------------


@dataclass(frozen=True)
class QuestionInstance:
    """One question filled in for a collection, with what a correct answer needs."""

    key: str
    kind: str
    """'aggregate', 'semantic' or 'unanswerable'."""
    question: str
    expected_route: Route
    bill_id: str | None
    """The bill holding the answer; None when the answer spans the collection."""
    evidence: str | None = None
    """Semantic: text the right chunk must contain."""
    expected: frozenset[str] = frozenset()
    """Aggregate: normalised values, any of which in the first result row is a match."""


Labels = Sequence[BillLabel]
Builder = Callable[[Labels], QuestionInstance | None]


def normalize_value(value: object) -> str:
    """Numbers rounded to cents; everything else case-folded and stripped."""
    text = str(value).strip()
    try:
        return str(Decimal(text).quantize(CENT, rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return text.casefold()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _pick(labels: Labels, predicate: Callable[[BillLabel], bool], n: int) -> BillLabel | None:
    """The n-th label (0-based, by id) matching the predicate; the last match if fewer."""
    matches = [label for label in labels if predicate(label)]
    return matches[min(n, len(matches) - 1)] if matches else None


def _credit(label: BillLabel) -> LineItem | None:
    return next((item for item in label.bill.line_items if item.amount < 0), None)


def _previous_balance(label: BillLabel) -> LineItem | None:
    return next((item for item in label.bill.line_items if "balance" in item.description.casefold()), None)


def _extra_charge(label: BillLabel) -> LineItem | None:
    return next((item for item in label.bill.line_items if item.description in EXTRA_CHARGES), None)


def _ids_of(bills: Sequence[BillLabel]) -> frozenset[str]:
    return frozenset(
        normalize_value(v) for label in bills for v in (label.id, label.bill.account_number, label.bill.provider)
    )


def _argbest(labels: Labels, key: Callable[[BillLabel], Decimal | None], highest: bool) -> list[BillLabel]:
    scored = [(key(label), label) for label in labels if key(label) is not None]
    if not scored:
        return []
    best = max(s for s, _ in scored) if highest else min(s for s, _ in scored)
    return [label for s, label in scored if s == best]


def _semantic(key: str, template: str, predicate: Callable[[BillLabel], bool], n: int,
              evidence: Callable[[BillLabel], str], kind: str = "semantic") -> Builder:
    def build(labels: Labels) -> QuestionInstance | None:
        label = _pick(labels, predicate, n)
        if label is None:
            return None
        values = {
            "provider": label.bill.provider,
            "account": label.bill.account_number,
            "credit": (_credit(label) or LineItem(description="", amount=Decimal(0))).description,
            "previous": (_previous_balance(label) or LineItem(description="", amount=Decimal(0))).description,
            "extra": (_extra_charge(label) or LineItem(description="", amount=Decimal(0))).description,
            "plan": label.bill.line_items[0].description,
        }
        return QuestionInstance(
            key=key,
            kind=kind,
            question=template.format(**values),
            expected_route=Route.SEMANTIC,
            bill_id=label.id,
            evidence=evidence(label) if kind == "semantic" else None,
        )

    return build


def _aggregate(key: str, question: str, expected: Callable[[Labels], tuple[frozenset[str], str | None]]) -> Builder:
    def build(labels: Labels) -> QuestionInstance | None:
        values, bill_id = expected(labels)
        if not values:
            return None
        return QuestionInstance(
            key=key, kind="aggregate", question=question, expected_route=Route.AGGREGATE, bill_id=bill_id, expected=values
        )

    return build


def _value(value: object) -> tuple[frozenset[str], None]:
    return frozenset({normalize_value(value)}), None


def _which(bills: list[BillLabel]) -> tuple[frozenset[str], str | None]:
    return _ids_of(bills), bills[0].id if len(bills) == 1 else None


def _average(values: Sequence[Decimal]) -> Decimal | None:
    return (sum(values, Decimal(0)) / len(values)).quantize(CENT, rounding=ROUND_HALF_UP) if values else None


def _kwh(label: BillLabel) -> Decimal | None:
    return label.bill.usage_amount if label.bill.usage_unit == "kWh" else None


def _template_is(name: str) -> Callable[[BillLabel], bool]:
    return lambda label: label.template == name


def _no_due_date_question(labels: Labels) -> QuestionInstance | None:
    """Two layouts print a note when there is no due date; use whichever the collection has."""
    for template, evidence in (("two_page", "No payment date applies to this bill"), ("two_column", "Upon receipt")):
        label = _pick(labels, lambda l, t=template: l.template == t and l.bill.due_date is None, 0)
        if label is not None:
            return QuestionInstance(
                key="S11 no due date",
                kind="semantic",
                question=f"What does the {label.bill.provider} bill say about when payment is due?",
                expected_route=Route.SEMANTIC,
                bill_id=label.id,
                evidence=evidence,
            )
    return None


def _desc(item: Callable[[BillLabel], LineItem | None]) -> Callable[[BillLabel], str]:
    return lambda label: (item(label) or LineItem(description="", amount=Decimal(0))).description


QUESTIONS: tuple[Builder, ...] = (
    # Semantic: the answer is text on one bill.
    _semantic("S1 credit", "What is the '{credit}' on the {provider} bill, and how much was it?",
              lambda l: _credit(l) is not None, 0, _desc(_credit)),
    _semantic("S2 extra charge", "What is the '{extra}' charge on account {account}, and how much is it?",
              lambda l: _extra_charge(l) is not None, 1, _desc(_extra_charge)),
    _semantic("S3 previous balance", "What is the '{previous}' line on the {provider} bill?",
              lambda l: _previous_balance(l) is not None, 0, _desc(_previous_balance)),
    _semantic("S4 CR notation", "What does 'CR' mean next to an amount on the {provider} bill?",
              _template_is("summary_box"), 0, lambda l: "CR denotes a credit"),
    _semantic("S5 parentheses", "What do amounts in parentheses mean on the {provider} statement?",
              _template_is("two_column"), 1, lambda l: "Amounts in parentheses are credits"),
    _semantic("S6 payment slip", "What does the payment slip on the {provider} bill ask me to do?",
              _template_is("two_page"), 0, lambda l: "return this portion with your payment"),
    _semantic("S7 tax line", "What is the tax line on account {account}?",
              lambda l: True, 3, lambda l: l.bill.line_items[-1].description),
    _semantic("S8 internet plan", "Describe the internet plan on the {provider} bill.",
              lambda l: l.bill.usage_amount is None, 0, lambda l: l.bill.line_items[0].description),
    _semantic("S9 late payments", "What does the {provider} bill say about late payments?",
              _template_is("summary_box"), 1, lambda l: "Late payments may be subject to interest"),
    _semantic("S10 contact", "Who should I contact with questions about account {account}?",
              _template_is("bordered_table"), 0, lambda l: "customer care"),
    _no_due_date_question,
    _semantic("S12 named line", "What is the '{extra}' line on the {provider} bill?",
              lambda l: _extra_charge(l) is not None, 4, _desc(_extra_charge)),
    _semantic("S13 balance carried", "What is the '{previous}' line on account {account}?",
              lambda l: _previous_balance(l) is not None, 2, _desc(_previous_balance)),
    # Unanswerable: nothing on the bill says this.
    _semantic("U1 phone number", "What is the customer's phone number on account {account}?",
              lambda l: True, 5, lambda l: "", kind="unanswerable"),
    _semantic("U2 early payment", "What discount would account {account} get for paying early?",
              lambda l: _credit(l) is None, 6, lambda l: "", kind="unanswerable"),
    # Aggregate: math or sorting across the collection.
    _aggregate("A1 highest total", "Which bill has the highest total amount due?",
               lambda ls: _which(_argbest(ls, lambda l: l.bill.total_amount, highest=True))),
    _aggregate("A2 lowest total", "Which bill has the lowest total amount due?",
               lambda ls: _which(_argbest(ls, lambda l: l.bill.total_amount, highest=False))),
    _aggregate("A3 sum USD", "What is the sum of the total amounts due on all USD bills?",
               lambda ls: _value(sum((l.bill.total_amount for l in ls if l.bill.currency == "USD"), Decimal(0)))),
    _aggregate("A4 average CAD", "What is the average total amount due across all CAD bills?",
               lambda ls: _value(_average([l.bill.total_amount for l in ls if l.bill.currency == "CAD"]))),
    _aggregate("A5 no due date", "How many bills have no due date?",
               lambda ls: _value(sum(1 for l in ls if l.bill.due_date is None))),
    _aggregate("A6 previous balance", "How many bills carry a balance forward from a previous bill?",
               lambda ls: _value(sum(1 for l in ls if _previous_balance(l) is not None))),
    _aggregate("A7 credits", "How many bills include a credit or discount (a negative line item)?",
               lambda ls: _value(sum(1 for l in ls if _credit(l) is not None))),
    _aggregate("A8 total kWh", "What is the total electricity usage in kWh across all bills?",
               lambda ls: _value(sum((k for l in ls if (k := _kwh(l)) is not None), Decimal(0)))),
    _aggregate("A9 most kWh", "Which bill used the most electricity in kWh?",
               lambda ls: _which(_argbest(ls, _kwh, highest=True))),
    _aggregate("A10 flat fee", "How many bills have no metered usage (flat-fee internet bills)?",
               lambda ls: _value(sum(1 for l in ls if l.bill.usage_amount is None))),
    _aggregate("A11 largest credit", "What is the largest single credit on any bill?",
               lambda ls: (frozenset({normalize_value(m), normalize_value(-m)}), None)
               if (m := min((i.amount for l in ls for i in l.bill.line_items), default=None)) is not None and m < 0
               else (frozenset(), None)),
    _aggregate("A12 CAD count", "How many bills are billed in CAD?",
               lambda ls: _value(sum(1 for l in ls if l.bill.currency == "CAD"))),
    _aggregate("A13 average kWh", "What is the average electricity usage per bill in kWh?",
               lambda ls: _value(_average([k for l in ls if (k := _kwh(l)) is not None]))),
    _aggregate("A14 combined 2024", "What is the combined amount due on all bills whose billing period started in 2024?",
               lambda ls: _value(sum((l.bill.total_amount for l in ls if l.bill.billing_period_start.year == 2024),
                                     Decimal(0)))),
    _aggregate("A15 average items", "What is the average number of line items per bill?",
               lambda ls: _value(_average([Decimal(len(l.bill.line_items)) for l in ls]))),
)


def build_questions(labels: Labels) -> tuple[list[QuestionInstance], int]:
    """Fill every template from the labels (sorted by id). Returns (questions, skipped count)."""
    ordered = sorted(labels, key=lambda label: label.id)
    built = [builder(ordered) for builder in QUESTIONS]
    return [q for q in built if q is not None], sum(q is None for q in built)


# --- scoring ------------------------------------------------------------------


def chunk_hit(answer: Answer, question: QuestionInstance, k: int = DEFAULT_TOP_K) -> bool:
    """A top-k chunk from the right bill contains the evidence text."""
    evidence = normalize_text(question.evidence or "")
    return any(
        chunk.bill_id == question.bill_id and evidence in normalize_text(chunk.text) for chunk in answer.chunks[:k]
    )


def bill_hit(answer: Answer, question: QuestionInstance, k: int = DEFAULT_TOP_K) -> bool:
    return any(chunk.bill_id == question.bill_id for chunk in answer.chunks[:k])


def aggregate_match(answer: Answer, question: QuestionInstance) -> bool:
    """The first result row contains an expected value."""
    if not answer.sql_rows:
        return False
    return any(normalize_value(cell) in question.expected for cell in answer.sql_rows[0])


class QuestionResult(BaseModel):
    key: str
    kind: str
    question: str
    bill_id: str | None
    expected_route: Route
    routed: Route
    route_method: str
    route_reason: str
    route_correct: bool
    chunk_hit: bool | None
    bill_hit: bool | None
    exact_match: bool | None
    expected_values: list[str]
    abstained: bool
    cited_bills: list[str]
    answer: Answer
    input_tokens: int
    output_tokens: int


class RagResults(BaseModel):
    split: str
    model: str
    bills: int
    questions: list[QuestionResult]
    skipped_questions: int
    cost_usd: Decimal | None


def evaluate_question(
    question: QuestionInstance,
    conn: psycopg.Connection,
    client: anthropic.Anthropic,
    config: ExtractionConfig,
    embedder: Embedder,
) -> QuestionResult:
    decision = route_question(question.question, client, config)
    answer = answer_question(question.question, conn, client, config, embedder, route_override=question.expected_route)
    is_semantic = question.kind == "semantic"
    return QuestionResult(
        key=question.key,
        kind=question.kind,
        question=question.question,
        bill_id=question.bill_id,
        expected_route=question.expected_route,
        routed=decision.route,
        route_method=decision.method,
        route_reason=decision.reason,
        route_correct=decision.route is question.expected_route,
        chunk_hit=chunk_hit(answer, question) if is_semantic else None,
        bill_hit=bill_hit(answer, question) if is_semantic else None,
        exact_match=aggregate_match(answer, question) if question.kind == "aggregate" else None,
        expected_values=sorted(question.expected),
        abstained=not answer.answerable,
        cited_bills=sorted({c.bill_id for c in answer.citations}),
        answer=answer,
        input_tokens=decision.input_tokens + answer.input_tokens,
        output_tokens=decision.output_tokens + answer.output_tokens,
    )


# --- report -------------------------------------------------------------------


def _rate(flags: Sequence[bool]) -> str:
    return f"{sum(flags)}/{len(flags)} ({100 * sum(flags) / len(flags):.0f}%)" if flags else "n/a"


def _cell(text: str, limit: int = 160) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_report(results: RagResults) -> str:
    qs = results.questions
    semantic = [q for q in qs if q.kind == "semantic"]
    aggregate = [q for q in qs if q.kind == "aggregate"]
    unanswerable = [q for q in qs if q.kind == "unanswerable"]
    summary = [
        ["Routing accuracy (all questions)", _rate([q.route_correct for q in qs])],
        ["  decided by rules", _rate([q.route_correct for q in qs if q.route_method == "rules"])],
        ["  decided by the model (ambiguous)", _rate([q.route_correct for q in qs if q.route_method != "rules"])],
        ["Recall@5, right chunk (semantic)", _rate([bool(q.chunk_hit) for q in semantic])],
        ["Recall@5, right bill (semantic)", _rate([bool(q.bill_hit) for q in semantic])],
        ["Exact match (aggregate)", _rate([bool(q.exact_match) for q in aggregate])],
        ["Answered and cited the right bill (semantic)",
         _rate([not q.abstained and q.bill_id in q.cited_bills for q in semantic])],
        ["Abstained (unanswerable)", _rate([q.abstained for q in unanswerable])],
        ["Citations dropped as not retrieved", str(sum(len(q.answer.dropped_citations) for q in qs))],
        ["LLM cost of the run", f"${results.cost_usd:.4f}" if results.cost_usd is not None else "unknown"],
    ]
    lines = [
        f"# RAG evaluation ({results.split} set)",
        "",
        f"{len(qs)} questions over {results.bills} ingested {results.split}-set bills. Model: `{results.model}`. "
        f"Embeddings: local sentence-transformers. Questions skipped (no matching bill): {results.skipped_questions}.",
        "Retrieval and SQL are measured on the labelled route; routing is measured separately. "
        "See the docstring of `src/rag/eval.py` for each measure's exact rule.",
        "",
        "## Summary",
        "",
        "| Measure | Result |",
        "|---|---|",
        *(f"| {name} | {value} |" for name, value in summary),
        "",
        "## Questions",
        "",
        "| Key | Question | Bill | Route (expected → routed, method) | Result | Answer |",
        "|---|---|---|---|---|---|",
    ]
    for q in qs:
        if q.kind == "semantic":
            result = f"chunk@5 {'✓' if q.chunk_hit else '✗'}, bill@5 {'✓' if q.bill_hit else '✗'}"
        elif q.kind == "aggregate":
            result = f"exact {'✓' if q.exact_match else '✗'} (expected {', '.join(q.expected_values[:3])})"
        else:
            result = f"abstained {'✓' if q.abstained else '✗'}"
        route = f"{q.expected_route} → {q.routed}{'' if q.route_correct else ' ✗'}, {q.route_method}"
        lines.append(
            f"| {q.key} | {_cell(q.question, 100)} | {q.bill_id or 'all'} | {route} | {result} | {_cell(q.answer.answer)} |"
        )
    return "\n".join(lines) + "\n"


def run_eval(
    split: str, conn: psycopg.Connection, client: anthropic.Anthropic, config: ExtractionConfig, embedder: Embedder,
    label_dir: Path = LABEL_DIR, log: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> RagResults:
    """Check the collection is the split, build the questions, and evaluate each."""
    split_ids = set(json.loads((label_dir / f"{split}_ids.json").read_text(encoding="utf-8")))
    ids = stored_bill_ids(conn)
    if not ids or not set(ids) <= split_ids:
        raise ValueError(f"the database does not hold the {split} collection; run: python -m src.storage.ingest --split {split}")
    labels = [BillLabel.model_validate_json((label_dir / f"{i}.json").read_text(encoding="utf-8")) for i in ids]
    questions, skipped = build_questions(labels)
    results: list[QuestionResult] = []
    for number, question in enumerate(questions, start=1):
        results.append(evaluate_question(question, conn, client, config, embedder))
        log(f"[{number}/{len(questions)}] {question.key}")
    tokens_in = sum(r.input_tokens for r in results)
    tokens_out = sum(r.output_tokens for r in results)
    return RagResults(
        split=split, model=config.model, bills=len(ids), questions=results, skipped_questions=skipped,
        cost_usd=cost_usd(config.model, tokens_in, tokens_out),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG question answering on the ingested collection.")
    parser.add_argument("--split", choices=sorted(RESULT_STEMS), required=True, help="the split that was ingested")
    args = parser.parse_args(argv)
    config = load_config()
    with connect(load_settings(), readonly=True) as conn:
        results = run_eval(args.split, conn, make_client(), config, load_embedder())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULT_STEMS[args.split]
    (RESULTS_DIR / f"{stem}.json").write_text(results.model_dump_json(indent=2), encoding="utf-8")
    report = render_report(results)
    (RESULTS_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
    print(report.split("## Questions")[0])
    print(f"Wrote {RESULTS_DIR / stem}.md and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
