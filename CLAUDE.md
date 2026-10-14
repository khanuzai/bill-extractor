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
    prompt.py        Instructions + output schema (derived from UtilityBill) + bill text
    client.py        Anthropic API call with structured output, retries, usage
    validate.py      Label-free consistency checks -> ValidationReport (pure, no API)
    extract.py       extract_bill(pdf_path) -> ExtractionResult; up to 2 correction retries
                     on failed checks; never raises on bad output
    cli.py           python -m src.extraction.cli <dev-set pdf>
  evaluation/    Grades extraction output against labels and parsers
    score.py         Scoring rules: per-field correct/missing/wrong, item-level line items
    cost.py          Token prices -> USD
    records.py       What a run records per bill
    run.py           python -m src.evaluation.run [--limit N]; test set only, LLM cache
    report.py        Markdown results tables
  storage/       PostgreSQL / pgvector persistence
    schema.sql       bills, line_items, chunks (vector(384), HNSW cosine index)
    db.py            Connections (owner and read-only role), schema setup, writes
    ingest.py        python -m src.storage.ingest --split dev|test; chunking strategy in docstring
  rag/           Question answering over the ingested bills
    embed.py         Local sentence-transformers embeddings (EMBEDDING_MODEL)
    router.py        aggregate vs semantic: keyword rules, model for ambiguous questions
    sql_answer.py    Model-written SQL, validated as one parameterized SELECT, run read-only
    retrieve.py      Top-k chunks by cosine similarity, scoped to bills the question names
    answer.py        python -m src.rag.answer "<question>"; grounded answer with citations
    eval.py          python -m src.rag.eval --split dev|test; 30 templated questions
  api/           FastAPI app
data/
  raw/           Source bill PDFs
  generated/     Output of the generator
  labels/        Ground-truth labels, plus dev_ids.json / test_ids.json
  results/       Evaluation output (committed): results.json/.md, llm_cache.json
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
- Extraction is developed on the dev set only. The extraction module, its prompt, and
  any tuning may only be developed against bills listed in `data/labels/dev_ids.json`.
  Never read, run on, or inspect test-set bills (`data/labels/test_ids.json`): not
  their PDFs, their extracted text, their labels, or the model's output for them.
- Tests never call the real Anthropic API; mock the client. Database tests use a throwaway
  `<POSTGRES_DB>_pytest` database and skip when Postgres is down.
- RAG question answering reads through the read-only role only. Generated SQL must pass
  `validate_select` and take every value as a bound parameter; never format values into SQL.
- The test set is scored once, by `src.evaluation.run` and `src.rag.eval --split test`.
  Never tune a prompt, check, question or parser on its results.
- Dates are `date` and money/quantities are `Decimal` — never strings or floats.
- Only add a dependency to `requirements.txt` when code actually uses it.
- Never read, print, or echo the contents of `.env`. Read config through `os.environ` only.
