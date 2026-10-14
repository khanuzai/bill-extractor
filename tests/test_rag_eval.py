"""The RAG evaluation's questions and scoring rules, on labels generated for the test run."""

from decimal import Decimal

import pytest

from src.generator.data import generate_bill, make_faker
from src.generator.render import TEMPLATES
from src.rag.answer import Answer, Citation
from src.rag.eval import (
    QUESTIONS,
    QuestionInstance,
    QuestionResult,
    RagResults,
    aggregate_match,
    bill_hit,
    build_questions,
    chunk_hit,
    normalize_value,
    render_report,
)
from src.rag.retrieve import RetrievedChunk
from src.rag.router import Route, route_question
from src.schema import BillLabel


@pytest.fixture(scope="module")
def labels() -> list[BillLabel]:
    """100 generated bills, rotating through the templates like the real data."""
    fake = make_faker(77)
    return [
        BillLabel(id=f"bill_{i:04d}", template=TEMPLATES[i % len(TEMPLATES)], bill=generate_bill(fake))
        for i in range(1, 101)
    ]


def _answer(chunks: list[RetrievedChunk] | None = None, rows: list[list[str]] | None = None) -> Answer:
    return Answer(
        question="q", route=Route.SEMANTIC, route_reason="", route_method="rules", answer="a", answerable=True,
        citations=[], dropped_citations=[], chunks=chunks or [], sql=None, sql_params=None, sql_columns=[],
        sql_rows=rows or [], sql_error=None, input_tokens=0, output_tokens=0,
    )


def test_there_are_30_questions_of_each_kind(labels: list[BillLabel]) -> None:
    questions, skipped = build_questions(labels)
    assert len(QUESTIONS) == 30 and skipped == 0 and len(questions) == 30
    kinds = [q.kind for q in questions]
    assert (kinds.count("aggregate"), kinds.count("semantic"), kinds.count("unanswerable")) == (15, 13, 2)
    assert len({q.key for q in questions}) == 30


def test_questions_name_a_real_bill_and_carry_what_scoring_needs(labels: list[BillLabel]) -> None:
    by_id = {label.id: label for label in labels}
    for q in build_questions(labels)[0]:
        if q.kind == "aggregate":
            assert q.expected and q.evidence is None
        else:
            assert q.bill_id in by_id
            bill = by_id[q.bill_id].bill
            assert bill.provider in q.question or bill.account_number in q.question
            assert (q.evidence is not None and q.evidence != "") == (q.kind == "semantic")


def test_questions_are_deterministic(labels: list[BillLabel]) -> None:
    assert build_questions(labels) == build_questions(list(reversed(labels)))


def test_expected_aggregate_values_come_from_the_labels(labels: list[BillLabel]) -> None:
    questions = {q.key: q for q in build_questions(labels)[0]}
    highest = max(labels, key=lambda label: label.bill.total_amount)
    assert normalize_value(highest.id) in questions["A1 highest total"].expected
    assert questions["A1 highest total"].bill_id == highest.id
    assert questions["A12 CAD count"].expected == {normalize_value(sum(1 for l in labels if l.bill.currency == "CAD"))}
    usd = sum((l.bill.total_amount for l in labels if l.bill.currency == "USD"), Decimal(0))
    assert questions["A3 sum USD"].expected == {normalize_value(usd)}


def test_rules_route_most_questions_and_agree_with_the_labels(labels: list[BillLabel]) -> None:
    for q in build_questions(labels)[0]:
        decision = route_question(q.question)
        if decision.method == "rules":
            assert decision.route is q.expected_route, q.question


@pytest.mark.parametrize(
    ("value", "expected"),
    [("412.3333333", "412.33"), (Decimal("12.5"), "12.50"), (7, "7.00"), ("  Bill_0003 ", "bill_0003"), ("-40.20", "-40.20")],
)
def test_normalize_value_rounds_numbers_and_folds_text(value: object, expected: str) -> None:
    assert normalize_value(value) == expected


def test_chunk_hit_needs_the_right_bill_and_the_evidence() -> None:
    question = QuestionInstance(
        key="k", kind="semantic", question="q", expected_route=Route.SEMANTIC, bill_id="bill_0003",
        evidence="CR denotes a credit",
    )
    right = RetrievedChunk("bill_0003", 1, 0, "h", "...  CR denotes a\ncredit. Late ...", 0.5)
    wrong_bill = RetrievedChunk("bill_0004", 1, 0, "h", "CR denotes a credit", 0.9)
    right_bill_other_text = RetrievedChunk("bill_0003", 1, 1, "h", "Energy charge 9.00", 0.8)

    assert chunk_hit(_answer([wrong_bill, right]), question)
    assert not chunk_hit(_answer([wrong_bill, right_bill_other_text]), question)
    assert bill_hit(_answer([wrong_bill, right_bill_other_text]), question)
    assert not chunk_hit(_answer([wrong_bill] * 5 + [right]), question), "only the top 5 count"


def test_aggregate_match_looks_at_the_first_row_only() -> None:
    question = QuestionInstance(
        key="k", kind="aggregate", question="q", expected_route=Route.AGGREGATE, bill_id=None,
        expected=frozenset({"412.33"}),
    )
    assert aggregate_match(_answer(rows=[["412.3333"]]), question)
    assert aggregate_match(_answer(rows=[["bill_0003", "412.33"]]), question)
    assert not aggregate_match(_answer(rows=[["1"], ["412.33"]]), question)
    assert not aggregate_match(_answer(rows=[]), question)


def test_report_summarises_every_measure() -> None:
    def result(kind: str, **flags: object) -> QuestionResult:
        return QuestionResult(
            key=kind, kind=kind, question="q?", bill_id="bill_0003", expected_route=Route.SEMANTIC, routed=Route.SEMANTIC,
            route_method="rules", route_reason="r", route_correct=True, chunk_hit=flags.get("chunk_hit"),
            bill_hit=flags.get("bill_hit"), exact_match=flags.get("exact_match"), expected_values=[],
            abstained=bool(flags.get("abstained", False)), cited_bills=["bill_0003"], answer=_answer(),
            input_tokens=10, output_tokens=5,
        )

    report = render_report(
        RagResults(
            split="dev", model="claude-haiku-4-5", bills=100, skipped_questions=0, cost_usd=Decimal("0.01"),
            questions=[
                result("semantic", chunk_hit=True, bill_hit=True),
                result("semantic", chunk_hit=False, bill_hit=True),
                result("aggregate", exact_match=True),
                result("unanswerable", abstained=True),
            ],
        )
    )
    assert "| Routing accuracy (all questions) | 4/4 (100%) |" in report
    assert "| Recall@5, right chunk (semantic) | 1/2 (50%) |" in report
    assert "| Exact match (aggregate) | 1/1 (100%) |" in report
    assert "| Abstained (unanswerable) | 1/1 (100%) |" in report
    assert Citation(bill_id="x", page=None).page is None
