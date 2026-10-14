"""SQL generation and validation for aggregate questions. Mocked model, no database."""

import re
from unittest.mock import MagicMock

import pytest
from fakes import message, mock_client

from src.extraction.client import ExtractionConfig
from src.schema import UtilityBill
from src.rag import sql_answer
from src.rag.sql_answer import (
    GeneratedQuery,
    QueryResult,
    SqlValidationError,
    answer_with_sql,
    generate_query,
    run_query,
    schema_for_prompt,
    validate_select,
)

CONFIG = ExtractionConfig(model="claude-haiku-4-5", effort=None)


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        ("SELECT id, total_amount FROM bills ORDER BY total_amount DESC LIMIT 1", {}),
        ("SELECT COUNT(*) FROM bills WHERE currency = %(currency)s", {"currency": "CAD"}),
        ("SELECT ROUND(AVG(total_amount), 2) FROM bills WHERE billing_period_start >= %(start)s::date", {"start": "2024-01-01"}),
        ("SELECT COUNT(DISTINCT bill_id) FROM line_items WHERE description ILIKE %(pattern)s", {"pattern": "%balance%"}),
        ("WITH t AS (SELECT bill_id, COUNT(*) AS n FROM line_items GROUP BY bill_id) SELECT ROUND(AVG(n), 2) FROM t", {}),
        ("SELECT b.id FROM bills b JOIN line_items li ON li.bill_id = b.id WHERE li.amount < 0", {}),
        ("SELECT id FROM public.bills WHERE provider ILIKE %(p)s AND currency = %(p2)s", {"p": "Acme%", "p2": "USD"}),
    ],
)
def test_safe_selects_are_accepted(sql: str, params: dict[str, str]) -> None:
    validate_select(sql, params)


@pytest.mark.parametrize(
    ("sql", "params", "reason"),
    [
        ("SELECT 1; DELETE FROM bills", {}, "exactly one statement"),
        ("DELETE FROM bills", {}, "expected a SELECT"),
        ("UPDATE bills SET total_amount = 0", {}, "expected a SELECT"),
        ("INSERT INTO bills (id) VALUES (1)", {}, "expected a SELECT"),
        ("DROP TABLE bills", {}, "expected a SELECT"),
        ("SELECT * INTO stolen FROM bills", {}, "Into"),
        ("WITH d AS (DELETE FROM bills RETURNING id) SELECT * FROM d", {}, "Delete"),
        ("SELECT id FROM bills FOR UPDATE", {}, "Lock"),
        ("SELECT id FROM bills WHERE currency = 'USD'", {}, "string literal"),
        ("SELECT id FROM bills WHERE provider = '' OR 1=1", {}, "string literal"),
        ("SELECT * FROM pg_user", {}, "table 'pg_user'"),
        ("SELECT text FROM chunks", {}, "table 'chunks'"),
        ("SELECT * FROM information_schema.tables", {}, "schema"),
        ("SELECT id FROM bills UNION SELECT usename FROM pg_user", {}, "expected a SELECT"),
        ("SELECT pg_sleep(10)", {}, "function"),
        ("SELECT id FROM bills -- comment", {}, "comments"),
        ("SELECT id FROM bills /* x */", {}, "comments"),
        ("SELECT id FROM bills WHERE currency = %(c)s", {}, "do not match"),
        ("SELECT id FROM bills", {"c": "USD"}, "do not match"),
        ("SELECT id FROM bills WHERE position % 2 = 0", {}, "'%'"),
        ("SET statement_timeout = 0", {}, "expected a SELECT"),
        ("COPY bills TO '/tmp/out'", {}, "expected a SELECT"),
        ("SELEC id FROM bills", {}, ""),
    ],
)
def test_unsafe_or_malformed_sql_is_rejected(sql: str, params: dict[str, str], reason: str) -> None:
    with pytest.raises(SqlValidationError, match=reason):
        validate_select(sql, params)


def test_prompt_schema_covers_every_column_of_the_queryable_tables_only() -> None:
    schema = schema_for_prompt()
    assert "CREATE TABLE IF NOT EXISTS bills" in schema and "CREATE TABLE IF NOT EXISTS line_items" in schema
    assert "chunks" not in schema and "INDEX" not in schema
    for column in [*UtilityBill.model_fields, "source_filename", "template", "position", "description", "amount"]:
        if column != "line_items":
            assert re.search(rf"^\s+{column}\s", schema, flags=re.MULTILINE), column
    assert schema.count("\n);") == 2


def test_generate_query_returns_sql_and_named_params() -> None:
    reply = {"sql": "SELECT COUNT(*) FROM bills WHERE currency = %(c)s;", "params": [{"name": "c", "value": "CAD"}]}
    client = mock_client(message(reply))

    query, error, tokens_in, tokens_out = generate_query(client, CONFIG, "How many CAD bills?")

    assert query == GeneratedQuery(sql="SELECT COUNT(*) FROM bills WHERE currency = %(c)s", params={"c": "CAD"})
    assert error is None and (tokens_in, tokens_out) == (100, 20)
    assert "How many CAD bills?" in client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_invalid_query_never_reaches_the_database() -> None:
    conn = MagicMock()
    with pytest.raises(SqlValidationError):
        run_query(conn, GeneratedQuery(sql="DELETE FROM bills", params={}))
    conn.cursor.assert_not_called()
    conn.transaction.assert_not_called()


def test_rejected_query_is_retried_once_with_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"sql": "SELECT id FROM bills WHERE currency = 'CAD'", "params": []}
    good = {"sql": "SELECT COUNT(*) FROM bills WHERE currency = %(c)s", "params": [{"name": "c", "value": "CAD"}]}
    client = mock_client(message(bad), message(good))
    result = QueryResult(columns=["count"], rows=[(12,)], truncated=False)
    monkeypatch.setattr(sql_answer, "run_query", lambda conn, query: (validate_select(query.sql, query.params), result)[1])

    answer = answer_with_sql(MagicMock(), client, CONFIG, "How many CAD bills?")

    assert answer.result == result and answer.error is None
    assert answer.query is not None and answer.query.params == {"c": "CAD"}
    retry_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "string literal 'CAD'" in retry_prompt
    assert (answer.input_tokens, answer.output_tokens) == (200, 40)


def test_failure_after_the_retry_is_reported_not_raised() -> None:
    bad = {"sql": "DROP TABLE bills", "params": []}
    answer = answer_with_sql(MagicMock(), mock_client(message(bad), message(bad)), CONFIG, "Drop it")
    assert answer.result is None
    assert answer.error is not None and "validation" in answer.error
