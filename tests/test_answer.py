"""The answer step: grounding, citations, and saying so when the answer isn't there."""

from unittest.mock import MagicMock

import pytest
from fakes import HashEmbedder, message, mock_client

from src.extraction.client import ExtractionConfig
from src.rag import answer as answer_module
from src.rag.answer import NOT_FOUND_MESSAGE, answer_question
from src.rag.retrieve import RetrievedChunk
from src.rag.router import Route
from src.rag.sql_answer import GeneratedQuery, QueryResult, SqlAnswer

CONFIG = ExtractionConfig(model="claude-haiku-4-5", effort=None)

CHUNKS = [
    RetrievedChunk("bill_0003", 1, 0, "Acme Power bill, account AC-1, page 1", "CR denotes a credit.", 0.71),
    RetrievedChunk("bill_0010", 2, 3, "Lake Gas bill, account 99, page 2", "Carbon levy  4.10", 0.55),
]


@pytest.fixture
def no_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer_module, "answer_with_sql", MagicMock(side_effect=AssertionError("SQL not expected")))


@pytest.fixture
def chunks(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(return_value=CHUNKS)
    monkeypatch.setattr(answer_module, "retrieve", fake)
    return fake


def test_semantic_answer_keeps_valid_citations_and_drops_invented_ones(no_sql: None, chunks: MagicMock) -> None:
    reply = {
        "answerable": True,
        "answer": "CR marks a credit [bill_0003 p.1].",
        "citations": [{"bill_id": "bill_0003", "page": 1}, {"bill_id": "bill_0003", "page": 2}, {"bill_id": "bill_9999", "page": 1}],
    }
    client = mock_client(message(reply))

    result = answer_question("What does 'CR' mean on the Acme Power bill?", MagicMock(), client, CONFIG, HashEmbedder())

    assert result.route is Route.SEMANTIC and result.route_method == "rules"
    assert result.answerable and result.answer.startswith("CR marks a credit")
    assert [(c.bill_id, c.page) for c in result.citations] == [("bill_0003", 1)]
    assert [(c.bill_id, c.page) for c in result.dropped_citations] == [("bill_0003", 2), ("bill_9999", 1)]
    assert result.chunks == CHUNKS
    context = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert '<source bill_id="bill_0003" page="1">' in context and "CR denotes a credit." in context
    assert "only the context" in client.messages.create.call_args.kwargs["system"]


def test_unanswerable_question_is_reported_as_not_found(no_sql: None, chunks: MagicMock) -> None:
    reply = {"answerable": False, "answer": "The bills provided don't list a phone number.", "citations": []}
    result = answer_question(
        "What is the customer's phone number on account AC-1?", MagicMock(), mock_client(message(reply)), CONFIG,
        HashEmbedder(), route_override=Route.SEMANTIC,
    )
    assert not result.answerable
    assert result.answer == "The bills provided don't list a phone number."
    assert result.citations == []


def test_empty_retrieval_answers_not_found_without_calling_the_model(no_sql: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer_module, "retrieve", MagicMock(return_value=[]))
    client = mock_client()
    result = answer_question("Why is there a fee?", MagicMock(), client, CONFIG, HashEmbedder())
    assert not result.answerable and result.answer == NOT_FOUND_MESSAGE
    client.messages.create.assert_not_called()


def test_aggregate_answer_reports_the_sql_result_and_cites_bills_in_it(
    chunks: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sql = SqlAnswer(
        query=GeneratedQuery("SELECT id, total_amount FROM bills ORDER BY total_amount DESC LIMIT 1", {}),
        result=QueryResult(columns=["id", "total_amount"], rows=[("bill_0042", 512.30)], truncated=False),
        error=None,
        input_tokens=100,
        output_tokens=20,
    )
    monkeypatch.setattr(answer_module, "answer_with_sql", MagicMock(return_value=sql))
    reply = {
        "answerable": True,
        "answer": "bill_0042 has the highest total, 512.3 [bill_0042].",
        "citations": [{"bill_id": "bill_0042", "page": None}, {"bill_id": "bill_0001", "page": None}],
    }
    client = mock_client(message(reply))

    result = answer_question("Which bill has the highest total amount due?", MagicMock(), client, CONFIG, HashEmbedder())

    assert result.route is Route.AGGREGATE
    chunks.assert_not_called()
    assert result.sql_rows == [["bill_0042", "512.3"]]
    assert [c.bill_id for c in result.citations] == ["bill_0042"]
    assert [c.bill_id for c in result.dropped_citations] == ["bill_0001"]
    assert (result.input_tokens, result.output_tokens) == (200, 40)
    assert "<sql_result>" in client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_failed_sql_answers_not_found_with_the_reason(chunks: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    failed = SqlAnswer(query=None, result=None, error="validation: Delete is not allowed", input_tokens=200, output_tokens=40)
    monkeypatch.setattr(answer_module, "answer_with_sql", MagicMock(return_value=failed))
    client = mock_client()

    result = answer_question("How many bills are in CAD?", MagicMock(), client, CONFIG, HashEmbedder())

    assert not result.answerable
    assert result.answer.startswith(NOT_FOUND_MESSAGE) and "Delete is not allowed" in result.answer
    assert result.sql_error == "validation: Delete is not allowed"
    client.messages.create.assert_not_called()


def test_route_override_skips_the_router(no_sql: None, chunks: MagicMock) -> None:
    reply = {"answerable": True, "answer": "x", "citations": []}
    client = mock_client(message(reply))
    result = answer_question("Tell me about it", MagicMock(), client, CONFIG, HashEmbedder(), route_override=Route.SEMANTIC)
    assert result.route_method == "override"
    assert client.messages.create.call_count == 1
