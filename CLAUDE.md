# bill-extractor

A pipeline that extracts structured data from utility bill PDFs using an LLM, then
grades that extraction against hand-written parsers on a labeled test set. Later it
adds RAG question-answering over the bills.

## Stack

- Python 3.11+
- Pydantic v2 (schema and validation)
- FastAPI (HTTP API)
- PostgreSQL with pgvector, run via Docker Compose
- PyMuPDF (PDF text extraction)
- Anthropic API (LLM extraction)
- Faker, Jinja2, WeasyPrint (synthetic bill generation; WeasyPrint needs the Pango
  system library — `brew install pango`)
- pytest

## Layout

```
src/
  schema.py      Pydantic models for a utility bill — the shared contract for every stage
  generator/     Generates synthetic bill PDFs and their labels
    data.py        Random UtilityBill data; its docstring defines the amount rule
    render.py      UtilityBill -> HTML template -> PDF
    templates/     Five bill layouts (Jinja2 HTML)
    cli.py         python -m src.generator.cli --count 200 --seed 42 --split
  parsers/       Hand-written parsers, used as the baseline to grade against
    text.py          PDF -> text with PyMuPDF
    regex_parser.py  Baseline regex parser -> ParseResult (fields are None when not found)
    README.md        Which templates the baseline targets, and why the rest are excluded
  extraction/    LLM-based extraction (PDF text -> schema)
  evaluation/    Grades extraction output against labels and parsers
  storage/       PostgreSQL / pgvector persistence
  api/           FastAPI app
data/
  raw/           Source bill PDFs
  generated/     Output of the generator
  labels/        Ground-truth labels for the test set
tests/           Unit tests
docker-compose.yml   pgvector/pgvector:pg16 with a named volume
.env.example         Template for the required env vars
decisions.md         Log of design decisions and why they were made
```

Imports are rooted at the repo: `from src.schema import UtilityBill`.

Run tests with `python -m pytest` from the repo root.

The generated `UtilityBill` is the ground-truth label: data is generated first and the
PDF is rendered from it. Never parse anything back out of a generated PDF to build a
label.

## Rules

- Type hints on everything: all function parameters, return types, and class attributes.
- No bare `except`. Catch the specific exception types you expect.
- No secrets in code. Read them from the environment via `python-dotenv`; document
  new variables in `.env.example`.
- Keep extraction and evaluation logic in plain functions that are easy to unit-test —
  no hidden I/O, no global state; pass inputs in and return results.
- Never edit files in `data/` directly. Data is only written by the pipeline's own code
  (e.g. the generator).
- The baseline regex parser targets only the `bordered_table` and `two_column`
  templates. Never add rules for, or tune it against, `summary_box`, `two_page` or
  `receipt` — they must stay unseen layouts (see `src/parsers/README.md`).
- Dates are `date` and money/quantities are `Decimal` — never strings or floats.
- Only add a dependency to `requirements.txt` when code actually uses it.
