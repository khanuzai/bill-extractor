"""Answer aggregate questions with a generated, validated, read-only SQL query.

The model writes one SELECT against the known schema. Every value taken from the
question (a provider name, a date, a currency) must be a named parameter, written
`%(name)s`, with the value returned separately; string literals in the SQL are
rejected, so nothing the user typed is ever interpolated into SQL text. The query is
validated as a single SELECT over the bill tables before it runs, and it runs as the
read-only database role, inside a read-only transaction, with a statement timeout.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import anthropic
import psycopg
import sqlglot
from pydantic import BaseModel
from sqlglot import exp

from src.extraction.client import ExtractionConfig
from src.rag.llm import ask
from src.storage.db import SCHEMA_PATH

ALLOWED_TABLES = frozenset({"bills", "line_items"})
MAX_ROWS = 50

# Expressions that write, change session state, lock rows or create objects.
FORBIDDEN_EXPRESSIONS: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Into,
    exp.Lock,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

_PLACEHOLDER = re.compile(r"%\((\w+)\)s")

SqlParam = str | int | float | bool | None


class SqlValidationError(ValueError):
    """The generated SQL is not a safe single SELECT."""


@dataclass(frozen=True)
class GeneratedQuery:
    sql: str
    params: dict[str, SqlParam]


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool


def schema_for_prompt() -> str:
    """The CREATE TABLE statements the model may query, with their column comments
    (chunks and the index omitted). Matched whole, up to the closing ");", because the
    comments themselves may contain semicolons."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    tables = re.findall(r"^CREATE TABLE IF NOT EXISTS (?:bills|line_items) \(.*?^\);", text, flags=re.MULTILINE | re.DOTALL)
    return "\n\n".join(tables)


SQL_SYSTEM_PROMPT = f"""\
You write one PostgreSQL SELECT query that answers a question about a collection of \
utility bills. The schema:

{schema_for_prompt()}

Rules:
- Exactly one SELECT statement (a WITH clause is fine). Read only. Tables: bills, \
line_items.
- Never put a value from the question in the SQL text. Write each one as a named \
parameter, %(name)s, and return its value in params. No quoted string literals at \
all; numeric literals (LIMIT 1, ROUND(x, 2)) are fine. Cast parameters where the type \
matters, e.g. %(start)s::date. To match text loosely, put the % wildcards in the \
parameter value and use ILIKE %(name)s.
- When the answer is a particular bill or bills, select bills.id (and any columns \
that help explain the answer), so the answer can cite them.
- Put the value that answers the question in the first column of the first row. \
Round averages to 2 decimal places.
- Do not use comments. Do not use the % operator.
"""


class _SqlParamReply(BaseModel):
    name: str
    value: str | int | float | bool | None


class _SqlReply(BaseModel):
    sql: str
    params: list[_SqlParamReply]


def validate_select(sql_text: str, params: dict[str, SqlParam]) -> None:
    """Raise SqlValidationError unless `sql_text` is a single safe SELECT.

    Checks: no comments; placeholders match `params` exactly; parses as exactly one
    statement whose root is a SELECT; nothing that writes, locks, sets or creates
    anywhere in the tree (including CTEs and SELECT INTO); only the bill tables (and
    the query's own CTEs); no string literals; no functions sqlglot does not recognise
    (which catches pg_sleep, dblink, pg_read_file and the like).
    """
    if "--" in sql_text or "/*" in sql_text:
        raise SqlValidationError("comments are not allowed")
    names = _PLACEHOLDER.findall(sql_text)
    if set(names) != set(params):
        raise SqlValidationError(
            f"placeholders {sorted(set(names))} do not match params {sorted(params)}"
        )
    numbered = sql_text
    for index, name in enumerate(dict.fromkeys(names), start=1):
        numbered = numbered.replace(f"%({name})s", f"${index}")
    if "%" in numbered:
        raise SqlValidationError("'%' is only allowed in %(name)s placeholders")

    try:
        statements = [s for s in sqlglot.parse(numbered, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SqlValidationError(f"could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise SqlValidationError(f"expected exactly one statement, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise SqlValidationError(f"expected a SELECT, got {type(tree).__name__}")
    for forbidden in FORBIDDEN_EXPRESSIONS:
        if tree.find(forbidden) is not None:
            raise SqlValidationError(f"{forbidden.__name__} is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.db and table.db.lower() != "public":
            raise SqlValidationError(f"schema {table.db!r} is not allowed")
        if table.name.lower() not in ALLOWED_TABLES | cte_names:
            raise SqlValidationError(f"table {table.name!r} is not allowed; use {sorted(ALLOWED_TABLES)}")
    for literal in tree.find_all(exp.Literal):
        if literal.is_string:
            raise SqlValidationError(
                f"string literal {literal.this!r} is not allowed; pass values as %(name)s parameters"
            )
    for function in tree.find_all(exp.Anonymous):
        raise SqlValidationError(f"function {function.name!r} is not allowed")


def generate_query(
    client: anthropic.Anthropic, config: ExtractionConfig, question: str, feedback: str | None = None
) -> tuple[GeneratedQuery | None, str | None, int, int]:
    """Ask the model for a query. Returns (query or None, error, input tokens, output tokens)."""
    user = f"<question>\n{question}\n</question>"
    if feedback:
        user += f"\n\nYour previous query failed:\n{feedback}\nWrite a corrected query."
    reply = ask(client, config, SQL_SYSTEM_PROMPT, user, _SqlReply)
    if reply.parsed is None:
        return None, reply.error, reply.input_tokens, reply.output_tokens
    params = {param.name: param.value for param in reply.parsed.params}
    return GeneratedQuery(sql=reply.parsed.sql.strip().rstrip(";"), params=params), None, reply.input_tokens, reply.output_tokens


def run_query(conn: psycopg.Connection, query: GeneratedQuery) -> QueryResult:
    """Validate and run the query in a read-only transaction; at most MAX_ROWS rows."""
    validate_select(query.sql, query.params)
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute(query.sql, query.params)  # values bound by the driver, never interpolated
        columns = [column.name for column in cursor.description or []]
        rows = cursor.fetchmany(MAX_ROWS + 1)
    return QueryResult(columns=columns, rows=[tuple(row) for row in rows[:MAX_ROWS]], truncated=len(rows) > MAX_ROWS)


@dataclass(frozen=True)
class SqlAnswer:
    query: GeneratedQuery | None
    result: QueryResult | None
    error: str | None
    input_tokens: int
    output_tokens: int


MAX_SQL_ATTEMPTS = 2


def answer_with_sql(
    conn: psycopg.Connection, client: anthropic.Anthropic, config: ExtractionConfig, question: str
) -> SqlAnswer:
    """Generate and run a query; on a validation or database error, retry once with the error."""
    feedback: str | None = None
    input_tokens = output_tokens = 0
    query: GeneratedQuery | None = None
    error: str | None = None
    for _ in range(MAX_SQL_ATTEMPTS):
        query, error, tokens_in, tokens_out = generate_query(client, config, question, feedback)
        input_tokens += tokens_in
        output_tokens += tokens_out
        if query is None:
            continue
        try:
            return SqlAnswer(query, run_query(conn, query), None, input_tokens, output_tokens)
        except SqlValidationError as exc:
            error = f"validation: {exc}"
        except psycopg.Error as exc:
            error = f"database: {exc}"
        feedback = f"{query.sql}\n{error}"
    return SqlAnswer(query, None, error, input_tokens, output_tokens)


def format_cell(value: object) -> str:
    """How a result value is shown to the model and in reports."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, date):
        return value.isoformat()
    return "NULL" if value is None else str(value)
