"""Retrieve the chunks most similar to a question, by cosine similarity in pgvector."""

from dataclasses import dataclass

import psycopg

from src.rag.embed import Embedder

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RetrievedChunk:
    bill_id: str
    page_number: int
    chunk_index: int
    heading: str
    text: str
    similarity: float
    """Cosine similarity to the question, -1 to 1."""


def named_bills(conn: psycopg.Connection, question: str) -> list[str]:
    """Ids of bills whose account number or provider name appears in the question.

    Case-insensitive substring match, done in SQL with the question as a bound
    parameter.
    """
    rows = conn.execute(
        """
        SELECT id FROM bills
        WHERE strpos(lower(%(q)s), lower(account_number)) > 0
           OR strpos(lower(%(q)s), lower(provider)) > 0
        ORDER BY id
        """,
        {"q": question},
    ).fetchall()
    return [row[0] for row in rows]


def retrieve(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    k: int = DEFAULT_TOP_K,
    scope_to_named_bills: bool = True,
) -> list[RetrievedChunk]:
    """The `k` chunks closest to the question, most similar first.

    Orders by pgvector's cosine distance (`<=>`), which the HNSW index on
    chunks.embedding serves. If the question names a bill by account number or
    provider, the search is limited to those bills first: embeddings match meaning
    well but exact identifiers poorly, so "account AC-33158-69" alone does not find
    its bill. A question that names no bill searches the whole collection.
    """
    query_vector = embedder.embed_query(question)
    scope = named_bills(conn, question) if scope_to_named_bills else []
    rows = conn.execute(
        """
        SELECT bill_id, page_number, chunk_index, heading, text, 1 - (embedding <=> %(q)s) AS similarity
        FROM chunks
        WHERE cardinality(%(scope)s::text[]) = 0 OR bill_id = ANY(%(scope)s::text[])
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": query_vector, "k": k, "scope": scope},
    ).fetchall()
    return [
        RetrievedChunk(
            bill_id=row[0], page_number=row[1], chunk_index=row[2], heading=row[3], text=row[4], similarity=float(row[5])
        )
        for row in rows
    ]
