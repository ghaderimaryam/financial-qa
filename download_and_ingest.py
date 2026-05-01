"""Download SEC HTML filings directly and ingest them.

Run once. Replaces the broken Tesla/NVIDIA PDF chunks with real text chunks.
"""
import urllib.request
from pathlib import Path

from src import config
from src.chunker import chunk_pages
from src.html_loader import load_html_filing
from src.vector_store import load_vectorstore


# SEC requires a User-Agent identifying you (free, no signup)
HEADERS = {"User-Agent": "Maryam Ghaderi maryam@example.com"}

FILINGS = {
    "Tesla_10K": "https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm",
    "NVIDIA_10K": "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm",
}

config.validate()
html_dir = config.DATA_DIR / "html"
html_dir.mkdir(parents=True, exist_ok=True)

vs = load_vectorstore()

# Step 1 — delete any existing Tesla/NVIDIA chunks (they're junk)
existing = vs.get()
ids_to_delete = [
    existing["ids"][i]
    for i, m in enumerate(existing["metadatas"])
    if m.get("source") in FILINGS
]
if ids_to_delete:
    vs.delete(ids=ids_to_delete)
    print(f"🗑️  Deleted {len(ids_to_delete)} junk chunks")

# Step 2 — download each filing as HTML, chunk, embed, add
for source_name, url in FILINGS.items():
    print(f"\n⬇️  Downloading {source_name}…")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        html_bytes = r.read()
    html_path = html_dir / f"{source_name}.html"
    html_path.write_bytes(html_bytes)
    print(f"   saved {len(html_bytes)//1024} KB to {html_path}")

    print(f"📄 Loading & chunking…")
    pages = load_html_filing(html_path, source_name=source_name)
    print(f"   {len(pages)} pages")

    docs = chunk_pages(pages)
    print(f"   {len(docs)} chunks")

    vs.add_documents(docs)
    print(f"   ✅ added to index")

# Final tally
from collections import Counter
final = vs.get()
print("\n" + "="*50)
print("Final chunk counts:")
for src, n in Counter(m.get("source") for m in final["metadatas"]).most_common():
    print(f"  {src}: {n}")
print(f"  Total: {len(final['metadatas'])}")