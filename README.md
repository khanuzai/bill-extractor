# bill-extractor

A pipeline that extracts structured data from utility bill PDFs using an LLM, then
grades that extraction against hand-written parsers on a labeled test set. Later it
adds RAG question-answering over the bills.

**Status:** skeleton only — schema and infrastructure are in place; the generator,
parsers, extraction, evaluation, storage, and API are not implemented yet.

## Setup

Requires Python 3.11+ and Docker.

```sh
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# then edit .env: set ANTHROPIC_API_KEY and a real POSTGRES_PASSWORD

# 3. Start PostgreSQL with pgvector
docker compose up -d
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
