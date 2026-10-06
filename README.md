# bill-extractor

A pipeline that extracts structured data from utility bill PDFs using an LLM, then
grades that extraction against hand-written parsers on a labeled test set. Later it
adds RAG question-answering over the bills.

**Status:** the schema, infrastructure, synthetic bill generator, baseline regex
parser, and LLM extraction are in place; the evaluation, storage, and API are not
implemented yet.

## Setup

Requires Python 3.11+ and Docker. WeasyPrint (PDF rendering) also needs the Pango
system library.

```sh
# 1. Install Pango, then create a virtual environment and install dependencies
brew install pango            # macOS; see the WeasyPrint docs for other platforms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# then edit .env: set ANTHROPIC_API_KEY and a real POSTGRES_PASSWORD

# 3. Start PostgreSQL with pgvector
docker compose up -d
```

## Generating the test set

```sh
python -m src.generator.cli --count 200 --seed 42 --split
```

Writes `data/generated/{id}.pdf` and `data/labels/{id}.json`, rotating evenly through
the five templates in `src/generator/templates/`. Each label holds the bill's id, the
template it was rendered with, and the `UtilityBill` the PDF was rendered from. With
`--split`, `data/labels/dev_ids.json` and `data/labels/test_ids.json` hold a 50/50 id
split balanced across templates. The same seed always produces identical files; each
run replaces the previous run's `bill_*` files.

## Extracting a bill by hand

```sh
python -m src.extraction.cli data/generated/<id>.pdf
```

Prints an `ExtractionResult` as JSON: the extracted bill (or `null` with an `error`),
token usage, and latency. Needs `ANTHROPIC_API_KEY` in `.env`. Use ids from
`data/labels/dev_ids.json` only; the CLI refuses test-set bills.

## Tests

```sh
python -m pytest
```

## Layout

```
src/schema.py     Pydantic models for a utility bill
src/generator/    Synthetic bill generation
src/parsers/      Hand-written baseline parsers
src/extraction/   LLM extraction
src/evaluation/   Grading against the labeled test set
src/storage/      PostgreSQL / pgvector persistence
src/api/          FastAPI app
data/             raw/, generated/, labels/
tests/            Unit tests
```
