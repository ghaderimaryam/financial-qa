"""Load SEC HTML filings directly. Bypasses the PDF print-to-PDF mess.

SEC HTML filings are well-structured and produce much cleaner text than
print-to-PDF, which on macOS sometimes flattens text to images.
"""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup


def load_html_filing(html_path: Path, source_name: str | None = None) -> list[dict]:
    """Return [{source, page, text}, ...] from a SEC HTML filing.

    SEC HTMLs don't have real page breaks, so we synthesize "pages" of ~3000
    characters each to keep citation granularity meaningful.
    """
    html_path = Path(html_path)
    source = source_name or html_path.stem
    raw = html_path.read_text(encoding="utf-8", errors="ignore")

    soup = BeautifulSoup(raw, "html.parser")

    # Strip script/style/nav noise
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Get text, preserve paragraph breaks
    text = soup.get_text(separator="\n")
    text = _normalize(text)

    # Synthetic pages — every ~3000 chars at a paragraph boundary
    pages: list[dict] = []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    buf = []
    page_num = 1
    char_count = 0
    for para in paragraphs:
        buf.append(para)
        char_count += len(para)
        if char_count >= 3000:
            pages.append({"source": source, "page": page_num, "text": "\n\n".join(buf)})
            buf = []
            char_count = 0
            page_num += 1
    if buf:
        pages.append({"source": source, "page": page_num, "text": "\n\n".join(buf)})

    return pages


def _normalize(text: str) -> str:
    text = text.replace("\xad", "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()