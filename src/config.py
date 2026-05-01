"""Project configuration. Loads from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = Path(os.getenv("DATA_DIR",     str(PROJECT_ROOT / "data")))
PDF_DIR      = Path(os.getenv("PDF_DIR",      str(DATA_DIR / "pdfs")))
CHROMA_PATH  = Path(os.getenv("CHROMA_PATH",  str(DATA_DIR / "chroma_db")))

# ─── Models ───────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL_GEN       = os.getenv("MODEL_GEN", "gpt-4o-mini")
MODEL_JUDGE     = os.getenv("MODEL_JUDGE", "gpt-4o-mini")  # for faithfulness checks
MODEL_EMBED     = os.getenv("MODEL_EMBED", "text-embedding-3-small")
COLLECTION_NAME = "financial_filings"

# ─── Retrieval ────────────────────────────────────────────────────────────
TOP_K           = int(os.getenv("TOP_K", "10"))

# ─── Chunking (semantic) ──────────────────────────────────────────────────
# Lower = more chunks, more granular. Higher = fewer chunks, broader context.
# 90th percentile of cosine distance is a sensible default for prose.
SEMANTIC_BREAKPOINT_PERCENTILE = 90

# ─── Faithfulness thresholds ─────────────────────────────────────────────
# score = (# of claims judged "yes") / (total claims)
FAITHFULNESS_BLOCK_BELOW = float(os.getenv("FAITHFULNESS_BLOCK_BELOW", "0.6"))
FAITHFULNESS_WARN_BELOW  = float(os.getenv("FAITHFULNESS_WARN_BELOW", "0.85"))


def validate() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
