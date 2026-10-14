"""Extract a split of bills and load them into PostgreSQL: structured rows and text chunks.

    python -m src.storage.ingest --split dev [--limit N]
    python -m src.storage.ingest --split test

The database holds one collection at a time: ingesting replaces whatever was there.
Extraction results are cached by bill id (the test split reuses the evaluation's
cache in data/results/llm_cache.json, so it costs nothing), and bills whose extraction
failed are skipped and reported.

Chunking strategy
-----------------
Each page is chunked separately, so every chunk has exactly one page number and a
citation can point at it.

1. Logical sections. The page text is split into blocks at blank lines. With
   PyMuPDF's sorted text, blank lines fall where the layout leaves vertical space:
   between the header, the address, a heading and its table, table rows, the footer.
   Runs of layout padding inside a line are collapsed to two spaces.
2. Packing. Consecutive blocks are joined into one chunk until adding the next block
   would pass MAX_CHUNK_CHARS. Blocks are never split here, so a table row or a
   footer note stays whole, and a heading usually lands with the rows under it.
3. Fixed-size fallback. A single block longer than MAX_CHUNK_CHARS (a layout with no
   blank lines, such as a dense receipt) is cut into windows of MAX_CHUNK_CHARS with
   OVERLAP_CHARS of overlap, breaking at whitespace, so a fact cut at one window's end
   appears whole at the next one's start.
4. Bill identity. Every chunk gets a heading: provider, account number, billing
   period and page. The heading is embedded with the text and shown to the answering
   model. Without it, a chunk such as "Delivery charge  13.11" is the same on every
   bill, and a question that names a provider or account could not find the right one.

Tradeoffs: blank-line blocks follow the layout only as well as the PDF's text
extraction preserves it; a layout that crams everything together falls back to
fixed windows, which can split a heading from its rows. Sizes are in characters, not
tokens, which is close enough for short English text. The heading comes from the LLM
extraction, so an extraction error (a wrong provider name) also misleads retrieval.
"""

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import anthropic
import psycopg
from pgvector.psycopg import register_vector

from src.evaluation.run import LLMCache, load_cache, save_cache
from src.extraction.client import load_config, make_client
from src.extraction.extract import ExtractionResult, extract_bill
from src.parsers.text import extract_pages
from src.rag.embed import Embedder, load_embedder
from src.schema import BillLabel, UtilityBill
from src.storage.db import (
    EMBEDDING_DIM,
    ChunkRow,
    StoredBill,
    clear_collection,
    connect,
    init_schema,
    load_settings,
    write_bill,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = REPO_ROOT / "data" / "labels"
PDF_DIR = REPO_ROOT / "data" / "generated"
RESULTS_DIR = REPO_ROOT / "data" / "results"
DEFAULT_CACHES: dict[str, Path] = {
    "dev": RESULTS_DIR / "llm_cache_dev.json",
    "test": RESULTS_DIR / "llm_cache.json",
}

MAX_CHUNK_CHARS = 350
OVERLAP_CHARS = 60

_BLANK_LINES = re.compile(r"\n[ \t]*\n")
_PADDING = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class Chunk:
    """A piece of one page's text."""

    page_number: int
    chunk_index: int
    text: str


def split_sections(page_text: str) -> list[str]:
    """Split a page into blank-line-separated blocks, with layout padding collapsed."""
    blocks = []
    for raw_block in _BLANK_LINES.split(page_text):
        lines = [_PADDING.sub("  ", line.strip()) for line in raw_block.splitlines()]
        block = "\n".join(line for line in lines if line)
        if block:
            blocks.append(block)
    return blocks


def split_fixed(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Cut text into windows of at most `max_chars`, overlapping by about `overlap`.

    Windows end at the last whitespace before the limit when there is one, and the
    next window starts at a whitespace boundary about `overlap` characters earlier.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start + overlap + 1, end)
            space = max(space, text.rfind("\n", start + overlap + 1, end))
            if space > start:
                end = space
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = end - overlap
        boundary = max(text.rfind(" ", start + 1, next_start), text.rfind("\n", start + 1, next_start))
        start = boundary + 1 if boundary > start else next_start
    return [window for window in windows if window]


def pack_sections(sections: Sequence[str], max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Join consecutive sections into chunks of at most `max_chars`, never splitting one
    unless it alone is too long (then it falls back to `split_fixed`)."""
    chunks: list[str] = []
    current = ""
    for section in sections:
        if len(section) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_fixed(section, max_chars))
        elif not current:
            current = section
        elif len(current) + 2 + len(section) <= max_chars:
            current = f"{current}\n\n{section}"
        else:
            chunks.append(current)
            current = section
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: Sequence[str], max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Chunk every page separately; chunk indexes run across the whole document."""
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for text in pack_sections(split_sections(page_text), max_chars):
            chunks.append(Chunk(page_number=page_number, chunk_index=len(chunks), text=text))
    return chunks


def chunk_heading(bill: UtilityBill, page_number: int) -> str:
    """The bill identity prepended to a chunk before embedding."""
    return (
        f"{bill.provider} bill, account {bill.account_number}, billing period "
        f"{bill.billing_period_start} to {bill.billing_period_end}, page {page_number}"
    )


def embedding_text(heading: str, text: str) -> str:
    return f"{heading}\n{text}"


def build_chunk_rows(bill: UtilityBill, pages: Sequence[str], embedder: Embedder) -> list[ChunkRow]:
    """Chunk a bill's pages and embed each chunk with its heading."""
    chunks = chunk_pages(pages)
    headings = [chunk_heading(bill, chunk.page_number) for chunk in chunks]
    vectors = embedder.embed_documents([embedding_text(h, c.text) for h, c in zip(headings, chunks, strict=True)])
    return [
        ChunkRow(page_number=c.page_number, chunk_index=c.chunk_index, heading=h, text=c.text, embedding=v)
        for c, h, v in zip(chunks, headings, vectors, strict=True)
    ]


def ingest(
    ids: Sequence[str],
    conn: psycopg.Connection,
    embedder: Embedder,
    extract: Callable[[Path], ExtractionResult],
    cache_path: Path,
    model: str,
    label_dir: Path = LABEL_DIR,
    pdf_dir: Path = PDF_DIR,
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> tuple[list[str], list[str]]:
    """Replace the collection with these bills. Returns (ingested ids, skipped ids).

    Commits once at the end, so a failed run leaves the previous collection intact.
    """
    if embedder.dimension != EMBEDDING_DIM:
        raise ValueError(
            f"embedding model produces {embedder.dimension}-dimensional vectors but the chunks table "
            f"stores {EMBEDDING_DIM}; change vector({EMBEDDING_DIM}) in schema.sql and db.EMBEDDING_DIM"
        )
    cache: LLMCache = load_cache(cache_path, model)
    clear_collection(conn)
    ingested: list[str] = []
    skipped: list[str] = []
    for number, bill_id in enumerate(ids, start=1):
        pdf_path = pdf_dir / f"{bill_id}.pdf"
        if bill_id not in cache.results:
            cache.results[bill_id] = extract(pdf_path)
            save_cache(cache_path, cache)
        result = cache.results[bill_id]
        if result.bill is None:
            skipped.append(bill_id)
            log(f"[{number}/{len(ids)}] {bill_id}: skipped, extraction failed: {result.error}")
            continue
        label = BillLabel.model_validate_json((label_dir / f"{bill_id}.json").read_text(encoding="utf-8"))
        stored = StoredBill(
            id=bill_id,
            source_filename=pdf_path.name,
            template=label.template,
            bill=result.bill,
            extraction_model=result.model,
        )
        write_bill(conn, stored, build_chunk_rows(result.bill, extract_pages(pdf_path), embedder))
        ingested.append(bill_id)
        log(f"[{number}/{len(ids)}] {bill_id}: ingested")
    conn.commit()
    return ingested, skipped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a split of bills and load them into PostgreSQL.")
    parser.add_argument("--split", choices=sorted(DEFAULT_CACHES), required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N bills of the split")
    parser.add_argument("--cache", type=Path, default=None, help="LLM result cache (default depends on --split)")
    args = parser.parse_args(argv)

    ids = sorted(json.loads((LABEL_DIR / f"{args.split}_ids.json").read_text(encoding="utf-8")))
    if args.limit is not None:
        ids = ids[: args.limit]
    config = load_config()
    client_holder: list[anthropic.Anthropic] = []

    def extract(pdf_path: Path) -> ExtractionResult:
        if not client_holder:
            client_holder.append(make_client())
        return extract_bill(pdf_path, client=client_holder[0], config=config)

    settings = load_settings()
    with connect(settings, register=False) as conn:
        init_schema(conn, settings)
        register_vector(conn)
        ingested, skipped = ingest(
            ids, conn, load_embedder(), extract, args.cache or DEFAULT_CACHES[args.split], config.model
        )
    print(f"Ingested {len(ingested)} {args.split} bills; skipped {len(skipped)}: {', '.join(skipped) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
