"""Page-level text extraction from PDFs using PyMuPDF (fitz).

Why PyMuPDF over pypdf: faster, better Unicode handling, and exposes per-page
text directly — critical for our citation system, which links every chunk back
to its page number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    """Return [{source, page, text}, ...] — one entry per page.

    `source` is the PDF's filename without extension (e.g. "Apple_10K").
    `page` is 1-indexed because that's what humans expect when citing.
    """
    pdf_path = Path(pdf_path)
    source = pdf_path.stem
    pages: list[dict] = []

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = _normalize(text)
            if text.strip():  # skip blank pages
                pages.append({"source": source, "page": i, "text": text})

    return pages


def load_all_pdfs(pdf_dir: Path) -> list[dict]:
    """Run `load_pdf_pages` over every .pdf in a directory."""
    pdf_dir = Path(pdf_dir)
    all_pages: list[dict] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        pages = load_pdf_pages(pdf)
        print(f"  • {pdf.name}: {len(pages)} pages")
        all_pages.extend(pages)
    return all_pages


def _normalize(text: str) -> str:
    """PDFs often have soft hyphens, weird spacing, ligatures. Tidy up."""
    text = text.replace("\xad", "")              # soft hyphens
    text = text.replace("\u00a0", " ")           # non-breaking spaces
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # ligatures
    # Collapse runs of whitespace but preserve paragraph breaks.
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
