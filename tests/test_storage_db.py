"""Integration tests against a real PostgreSQL with pgvector (docker compose up -d).

They run in a throwaway database, `<POSTGRES_DB>_pytest`, with a throwaway read-only
role, so the real collection is never touched. Skipped when Postgres is unreachable.
"""

import secrets
from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from decimal import Decimal

import psycopg
import pytest
from fakes import HashEmbedder
from psycopg import sql
from pgvector.psycopg import register_vector

from src.rag.retrieve import named_bills, retrieve
from src.rag.sql_answer import GeneratedQuery, SqlValidationError, run_query
from src.schema import LineItem, UtilityBill
from src.storage.db import DbSettings, StoredBill, connect, init_schema, load_settings, write_bill
from src.storage.ingest import build_chunk_rows


def _admin_settings() -> DbSettings | None:
    try:
        settings = load_settings()
        with psycopg.connect(
            host=settings.host, port=settings.port, dbname=settings.dbname, user=settings.user,
            password=settings.password, connect_timeout=2,
        ):
            return settings
    except (KeyError, psycopg.OperationalError):
        return None


ADMIN = _admin_settings()
pytestmark = pytest.mark.skipif(ADMIN is None, reason="PostgreSQL is not reachable (docker compose up -d)")


def make_bill(provider: str, total: str, items: list[tuple[str, str]], currency: str = "USD") -> UtilityBill:
    return UtilityBill(
        provider=provider, account_number=f"AC-{provider[:3].upper()}", billing_period_start=date(2025, 1, 1),
        billing_period_end=date(2025, 1, 31), due_date=None, usage_amount=Decimal("300"), usage_unit="kWh",
        line_items=[LineItem(description=d, amount=Decimal(a)) for d, a in items], total_amount=Decimal(total),
        currency=currency,
    )


@pytest.fixture(scope="module")
def settings() -> Iterator[DbSettings]:
    assert ADMIN is not None
    test_settings = replace(
        ADMIN, dbname=f"{ADMIN.dbname}_pytest", readonly_user="bills_reader_pytest",
        readonly_password=secrets.token_urlsafe(16),
    )
    with psycopg.connect(
        host=ADMIN.host, port=ADMIN.port, dbname=ADMIN.dbname, user=ADMIN.user, password=ADMIN.password, autocommit=True
    ) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_settings.dbname)))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_settings.dbname)))
    with connect(test_settings, register=False) as conn:
        init_schema(conn, test_settings)
        register_vector(conn)
        embedder = HashEmbedder()
        for bill_id, bill, pages in [
            ("bill_0001", make_bill("Acme Power", "110.00", [("Energy charge", "100.00"), ("Sales tax", "10.00")]),
             ["Acme Power\n\nEnergy charge  100.00\n\nSales tax  10.00"]),
            ("bill_0002", make_bill("Lake Gas", "45.00", [("Gas supply", "50.00"), ("Loyalty credit", "-5.00")], "CAD"),
             ["Lake Gas\n\nGas supply  50.00", "Loyalty credit  (5.00)\n\nAmounts in parentheses are credits."]),
        ]:
            stored = StoredBill(id=bill_id, source_filename=f"{bill_id}.pdf", template=None, bill=bill, extraction_model="m")
            write_bill(conn, stored, build_chunk_rows(bill, pages, embedder))
        conn.commit()
    yield test_settings
    with psycopg.connect(
        host=ADMIN.host, port=ADMIN.port, dbname=ADMIN.dbname, user=ADMIN.user, password=ADMIN.password, autocommit=True
    ) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(test_settings.dbname)))
        admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(test_settings.readonly_user)))


@pytest.fixture
def reader(settings: DbSettings) -> Iterator[psycopg.Connection]:
    with connect(settings, readonly=True) as conn:
        yield conn


def test_rows_and_chunks_are_stored(settings: DbSettings) -> None:
    with connect(settings) as conn:
        assert conn.execute("SELECT id, total_amount, currency FROM bills ORDER BY id").fetchall() == [
            ("bill_0001", Decimal("110.00"), "USD"),
            ("bill_0002", Decimal("45.00"), "CAD"),
        ]
        assert conn.execute("SELECT position, amount FROM line_items WHERE bill_id = 'bill_0002' ORDER BY 1").fetchall() == [
            (1, Decimal("50.00")),
            (2, Decimal("-5.00")),
        ]
        pages = conn.execute("SELECT page_number FROM chunks WHERE bill_id = 'bill_0002' ORDER BY chunk_index").fetchall()
        assert [p[0] for p in pages] == [1, 2]


def test_rewriting_a_bill_replaces_its_rows(settings: DbSettings) -> None:
    with connect(settings) as conn:
        bill = make_bill("Acme Power", "110.00", [("Energy charge", "100.00"), ("Sales tax", "10.00")])
        stored = StoredBill(id="bill_0001", source_filename="bill_0001.pdf", template=None, bill=bill, extraction_model="m")
        write_bill(conn, stored, build_chunk_rows(bill, ["Acme Power"], HashEmbedder()))
        assert conn.execute("SELECT count(*) FROM line_items WHERE bill_id = 'bill_0001'").fetchone() == (2,)
        conn.rollback()


def test_hnsw_cosine_index_exists(reader: psycopg.Connection) -> None:
    definition = reader.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'").fetchone()
    assert definition is not None and "hnsw" in definition[0] and "vector_cosine_ops" in definition[0]


def test_retrieve_returns_the_most_similar_chunk_with_bill_and_page(reader: psycopg.Connection) -> None:
    chunks = retrieve(reader, HashEmbedder(), "What do amounts in parentheses mean?", k=2)
    assert (chunks[0].bill_id, chunks[0].page_number) == ("bill_0002", 2)
    assert "parentheses" in chunks[0].text
    assert chunks[0].similarity >= chunks[1].similarity


@pytest.mark.parametrize(
    "statement",
    ["DELETE FROM bills", "UPDATE bills SET total_amount = 0", "CREATE TABLE evil (x int)", "DROP TABLE chunks"],
)
def test_read_only_role_cannot_write(reader: psycopg.Connection, statement: str) -> None:
    with pytest.raises((psycopg.errors.ReadOnlySqlTransaction, psycopg.errors.InsufficientPrivilege)):
        reader.execute(statement)
    reader.rollback()


def test_read_only_role_cannot_write_even_outside_a_read_only_transaction(settings: DbSettings) -> None:
    with psycopg.connect(
        host=settings.host, port=settings.port, dbname=settings.dbname, user=settings.readonly_user,
        password=settings.readonly_password,
    ) as conn:
        conn.execute("SET TRANSACTION READ WRITE")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM bills")


def test_validated_query_runs_with_bound_parameters(reader: psycopg.Connection) -> None:
    query = GeneratedQuery(sql="SELECT id, total_amount FROM bills WHERE currency = %(c)s", params={"c": "CAD"})
    result = run_query(reader, query)
    assert result.columns == ["id", "total_amount"]
    assert result.rows == [("bill_0002", Decimal("45.00"))]


def test_injection_in_a_parameter_is_just_a_value(reader: psycopg.Connection) -> None:
    query = GeneratedQuery(sql="SELECT count(*) FROM bills WHERE provider = %(p)s", params={"p": "x' OR '1'='1"})
    assert run_query(reader, query).rows == [(0,)]


def test_unsafe_query_is_refused_before_execution(reader: psycopg.Connection) -> None:
    with pytest.raises(SqlValidationError):
        run_query(reader, GeneratedQuery(sql="DELETE FROM bills", params={}))
    assert reader.execute("SELECT count(*) FROM bills").fetchone() == (2,)


def test_question_naming_a_bill_is_scoped_to_it(reader: psycopg.Connection) -> None:
    assert named_bills(reader, "What is the sales tax on account ac-acm?") == ["bill_0001"]
    assert named_bills(reader, "What does LAKE GAS charge?") == ["bill_0002"]
    assert named_bills(reader, "What do amounts in parentheses mean?") == []

    chunks = retrieve(reader, HashEmbedder(), "What do amounts in parentheses mean on the Acme Power bill?", k=5)
    assert chunks and {c.bill_id for c in chunks} == {"bill_0001"}

    unscoped = retrieve(reader, HashEmbedder(), "What do amounts in parentheses mean on the Acme Power bill?", k=5,
                        scope_to_named_bills=False)
    assert {c.bill_id for c in unscoped} == {"bill_0001", "bill_0002"}
