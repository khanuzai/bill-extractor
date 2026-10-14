"""Chunking and chunk-row building for ingestion. No database, no API."""

from pathlib import Path

import pytest
from fakes import HashEmbedder

from src.generator.data import generate_bill, generate_extras, make_faker
from src.generator.render import TEMPLATES, render_pdf
from src.parsers.text import extract_pages
from src.storage.ingest import (
    MAX_CHUNK_CHARS,
    build_chunk_rows,
    chunk_heading,
    chunk_pages,
    pack_sections,
    split_fixed,
    split_sections,
)


def test_split_sections_breaks_at_blank_lines_and_collapses_padding() -> None:
    page = "Acme Power      Account Number\n      0123\n\n\nCHARGES\n\n Energy charge            95.12\n   \n Tax   5.00\n"
    assert split_sections(page) == ["Acme Power  Account Number\n0123", "CHARGES", "Energy charge  95.12", "Tax  5.00"]


def test_split_sections_of_an_empty_page_is_empty() -> None:
    assert split_sections("  \n\n \n") == []


def test_pack_sections_joins_neighbours_up_to_the_limit_without_splitting_them() -> None:
    sections = ["a" * 40, "b" * 40, "c" * 40]
    assert pack_sections(sections, max_chars=90) == ["a" * 40 + "\n\n" + "b" * 40, "c" * 40]


def test_pack_sections_falls_back_to_fixed_windows_for_one_long_section() -> None:
    long_section = " ".join(f"word{i}" for i in range(60))
    chunks = pack_sections(["short", long_section, "tail"], max_chars=100)
    assert chunks[0] == "short"
    assert chunks[-1] == "tail"
    assert len(chunks) > 3
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_split_fixed_windows_overlap_and_cover_the_text() -> None:
    text = " ".join(f"w{i:03d}" for i in range(200))
    windows = split_fixed(text, max_chars=120, overlap=30)
    assert all(len(w) <= 120 for w in windows)
    assert windows[0].startswith("w000") and windows[-1].endswith("w199")
    for left, right in zip(windows, windows[1:]):
        assert right.split()[0] in left.split(), "consecutive windows share words"
    words = {word for window in windows for word in window.split()}
    assert words == set(text.split())


def test_split_fixed_never_cuts_inside_a_word_when_there_is_whitespace() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    for window in split_fixed(text, max_chars=20, overlap=5):
        assert all(word in text.split() for word in window.split())


def test_split_fixed_rejects_overlap_not_smaller_than_window() -> None:
    with pytest.raises(ValueError):
        split_fixed("text", max_chars=10, overlap=10)


def test_chunks_never_cross_pages_and_carry_their_page_number() -> None:
    chunks = chunk_pages(["page one header\n\npage one body", "page two"], max_chars=1000)
    assert [(c.page_number, c.text) for c in chunks] == [(1, "page one header\n\npage one body"), (2, "page two")]
    assert [c.chunk_index for c in chunks] == [0, 1]


@pytest.mark.parametrize("template", TEMPLATES)
def test_real_bill_layouts_chunk_within_the_limit_and_keep_every_line(template: str, tmp_path: Path) -> None:
    fake = make_faker(31)
    bill = generate_bill(fake)
    pdf = tmp_path / "bill.pdf"
    render_pdf(bill, generate_extras(fake, bill), template, pdf)
    pages = extract_pages(pdf)

    chunks = chunk_pages(pages)

    assert chunks and all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert {c.page_number for c in chunks} == set(range(1, len(pages) + 1))
    chunk_text = " ".join(" ".join(c.text.split()) for c in chunks)
    for page in pages:
        for line in page.splitlines():
            if line.strip():
                assert " ".join(line.split()) in chunk_text


def test_chunk_rows_embed_heading_with_text(tmp_path: Path) -> None:
    fake = make_faker(32)
    bill = generate_bill(fake)
    pdf = tmp_path / "bill.pdf"
    render_pdf(bill, generate_extras(fake, bill), "two_page", pdf)
    embedder = HashEmbedder()

    rows = build_chunk_rows(bill, extract_pages(pdf), embedder)

    assert {row.page_number for row in rows} == {1, 2}
    for row in rows:
        assert row.heading == chunk_heading(bill, row.page_number)
        assert bill.provider in row.heading and bill.account_number in row.heading
        assert row.embedding.shape == (384,)
        assert row.embedding.tolist() == embedder.embed_query(f"{row.heading}\n{row.text}").tolist()
