"""Build the vector index from PDFs in data/pdfs/.

Usage:
    python ingest.py                          # ingest everything in data/pdfs/
    python ingest.py --pdf path/to/file.pdf   # ingest one specific PDF (full rebuild)
    python ingest.py --add path/to/file.pdf   # add to existing index without rebuilding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config
from src.chunker import chunk_pages
from src.pdf_loader import load_all_pdfs, load_pdf_pages
from src.vector_store import build_vectorstore, load_vectorstore


SAMPLE_INSTRUCTIONS = """
ℹ️  No PDFs found in data/pdfs/. To get started:

  Option A — bundled samples (recommended for first run):
    1. Apple 10-K (FY2023):
       https://www.apple.com/newsroom/pdfs/fy2023-q4/_10-K-2023-As-Filed.pdf
       Save as data/pdfs/Apple_10K.pdf

    2. Tesla 10-K:
       https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001318605&type=10-K
       Click the most recent 10-K → click the .htm file → File > Print > Save as PDF
       Save as data/pdfs/Tesla_10K.pdf

  Option B — your own filing:
    Drop any 10-K (or other) PDF into data/pdfs/ and re-run.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vector index from PDFs.")
    parser.add_argument("--pdf", type=str, help="Ingest one PDF (full rebuild).")
    parser.add_argument("--add", type=str, help="Add a PDF to existing index without rebuild.")
    args = parser.parse_args()

    config.validate()
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)

    # ── Append mode (used by Gradio upload button) ─────────────────────
    if args.add:
        pdf_path = Path(args.add)
        if not pdf_path.exists():
            sys.exit(f"❌ Not found: {pdf_path}")
        print(f"📄 Adding {pdf_path.name} to existing index…")
        pages = load_pdf_pages(pdf_path)
        print(f"   {len(pages)} pages")
        chunks = chunk_pages(pages)
        print(f"   {len(chunks)} chunks")
        vs = load_vectorstore()
        vs.add_documents(chunks)
        print(f"✅ Index now has {vs._collection.count()} chunks")
        return

    # ── Full rebuild path ──────────────────────────────────────────────
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            sys.exit(f"❌ Not found: {pdf_path}")
        print(f"📄 Loading {pdf_path.name}…")
        pages = load_pdf_pages(pdf_path)
    else:
        if not any(config.PDF_DIR.glob("*.pdf")):
            print(SAMPLE_INSTRUCTIONS)
            sys.exit(1)
        print(f"📄 Loading PDFs from {config.PDF_DIR}…")
        pages = load_all_pdfs(config.PDF_DIR)

    if not pages:
        sys.exit("❌ No usable text extracted. Are the PDFs scanned images?")

    print(f"   {len(pages)} total pages")
    print("✂️  Chunking semantically (this calls the embedding API)…")
    chunks = chunk_pages(pages)
    print(f"   {len(chunks)} chunks produced")

    print("🗄️  Building vector store…")
    build_vectorstore(chunks, rebuild=True)
    print("\n✅ Index built. Run `python app.py` to launch the UI.")


if __name__ == "__main__":
    main()
