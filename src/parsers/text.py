"""Extract plain text from a bill PDF with PyMuPDF."""

from pathlib import Path

import pymupdf

PAGE_SEPARATOR = "\n"


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the text of each page of the PDF, in page order.

    Within a page, text is ordered top to bottom then left to right, so a label and
    the value printed beside it land on the same line, separated by a run of spaces.
    """
    with pymupdf.open(pdf_path) as document:
        return [page.get_text("text", sort=True) for page in document]


def extract_text(pdf_path: Path) -> str:
    """Return the text of every page of the PDF, in reading order, pages concatenated."""
    return PAGE_SEPARATOR.join(extract_pages(pdf_path))
