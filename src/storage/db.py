"""PostgreSQL access: schema setup, writes for ingestion, read-only connections for RAG.

Two roles are used. The owner (POSTGRES_USER) creates the schema and writes bills
and chunks. Question answering connects as a separate read-only role
(POSTGRES_READONLY_USER), which can only SELECT, defaults every transaction to read
only, and has a statement timeout; `connect(readonly=True)` also opens its
connection in read-only mode, so a write is refused at two levels.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql

from src.schema import UtilityBill

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
EMBEDDING_DIM = 384  # must match vector(384) in schema.sql
READONLY_STATEMENT_TIMEOUT_MS = 5000
TABLES: tuple[str, ...] = ("bills", "line_items", "chunks")


@dataclass(frozen=True)
class DbSettings:
    """Connection settings, read from the environment."""

    host: str
    port: int
    dbname: str
    user: str
    password: str
    readonly_user: str
    readonly_password: str | None


@dataclass(frozen=True)
class StoredBill:
    """One extracted bill and where it came from."""

    id: str
    source_filename: str
    template: str | None
    bill: UtilityBill
    extraction_model: str


@dataclass(frozen=True)
class ChunkRow:
    """One text chunk with its embedding, ready to insert."""

    page_number: int
    chunk_index: int
    heading: str
    text: str
    embedding: np.ndarray


def load_settings() -> DbSettings:
    """Read connection settings from the environment (and .env)."""
    load_dotenv()
    return DbSettings(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        readonly_user=os.environ.get("POSTGRES_READONLY_USER", "bills_reader"),
        readonly_password=os.environ.get("POSTGRES_READONLY_PASSWORD") or None,
    )


def connect(settings: DbSettings, readonly: bool = False, register: bool = True) -> psycopg.Connection:
    """Open a connection as the owner, or as the read-only role.

    `register` adapts numpy arrays to pgvector; pass False before the schema (and its
    `vector` extension) exists.
    """
    if readonly:
        if settings.readonly_password is None:
            raise RuntimeError("POSTGRES_READONLY_PASSWORD is not set; add it to .env (see .env.example)")
        conn = psycopg.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.readonly_user,
            password=settings.readonly_password,
            options=f"-c statement_timeout={READONLY_STATEMENT_TIMEOUT_MS}",
        )
        conn.read_only = True
    else:
        conn = psycopg.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.user,
            password=settings.password,
        )
    if register:
        register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, settings: DbSettings) -> None:
    """Create the tables, index and read-only role (idempotent), then commit."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    if settings.readonly_password is None:
        raise RuntimeError("POSTGRES_READONLY_PASSWORD is not set; add it to .env (see .env.example)")
    ensure_readonly_role(conn, settings.readonly_user, settings.readonly_password, settings.dbname)
    conn.commit()


def ensure_readonly_role(conn: psycopg.Connection, role: str, password: str, dbname: str) -> None:
    """Create or update a login role that can only read the bill tables."""
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone() is not None
    name = sql.Identifier(role)
    verb = sql.SQL("ALTER") if exists else sql.SQL("CREATE")
    conn.execute(sql.SQL("{} ROLE {} WITH LOGIN PASSWORD {}").format(verb, name, sql.Literal(password)))
    conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(name))
    conn.execute(
        sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(name, sql.Literal(READONLY_STATEMENT_TIMEOUT_MS))
    )
    conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(dbname), name))
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(name))
    conn.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(name))
    tables = sql.SQL(", ").join(sql.Identifier(table) for table in TABLES)
    conn.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(tables, name))


def clear_collection(conn: psycopg.Connection) -> None:
    """Delete every bill, line item and chunk."""
    conn.execute("TRUNCATE bills, line_items, chunks")


def write_bill(conn: psycopg.Connection, stored: StoredBill, chunks: Sequence[ChunkRow]) -> None:
    """Insert (or replace) one bill with its line items and chunks."""
    bill = stored.bill
    conn.execute("DELETE FROM bills WHERE id = %s", (stored.id,))
    conn.execute(
        """
        INSERT INTO bills (id, source_filename, template, provider, account_number,
                           billing_period_start, billing_period_end, due_date,
                           usage_amount, usage_unit, total_amount, currency, extraction_model)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            stored.id,
            stored.source_filename,
            stored.template,
            bill.provider,
            bill.account_number,
            bill.billing_period_start,
            bill.billing_period_end,
            bill.due_date,
            bill.usage_amount,
            bill.usage_unit,
            bill.total_amount,
            bill.currency,
            stored.extraction_model,
        ),
    )
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO line_items (bill_id, position, description, amount) VALUES (%s, %s, %s, %s)",
            [(stored.id, position, item.description, item.amount) for position, item in enumerate(bill.line_items, 1)],
        )
        cursor.executemany(
            "INSERT INTO chunks (bill_id, page_number, chunk_index, heading, text, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            [(stored.id, c.page_number, c.chunk_index, c.heading, c.text, c.embedding) for c in chunks],
        )


def stored_bill_ids(conn: psycopg.Connection) -> list[str]:
    """Ids of every bill in the collection, sorted."""
    return [row[0] for row in conn.execute("SELECT id FROM bills ORDER BY id").fetchall()]

