"""Extract plain text from a bill PDF with PyMuPDF."""

from pathlib import Path

import pymupdf

PAGE_SEPARATOR = "\n"


def extract_text(pdf_path: Path) -> str:
    """Return the text of every page of the PDF, in reading order.

    Within a page, text is ordered top to bottom then left to right, so a label and
    the value printed beside it land on the same line, separated by a run of spaces.
    Pages are concatenated in order.
    """
    with pymupdf.open(pdf_path) as document:
        pages = [page.get_text("text", sort=True) for page in document]
    return PAGE_SEPARATOR.join(pages)
