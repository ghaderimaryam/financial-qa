"""Semantic chunking: split text where consecutive sentences become semantically distant.

Standard fixed-size splitting (e.g. 1000 chars + 200 overlap) is wrong for financial
docs — it slices a single risk-factor disclosure across two chunks, which destroys
retrieval quality. Semantic chunking embeds each sentence and starts a new chunk
when the cosine distance between adjacent sentences exceeds a percentile threshold.
The result: chunks end at natural section/topic boundaries.

We process page-by-page so chunks never span pages — keeping page-level citation
exact. Each output chunk carries (source, page, text) metadata.
"""
from __future__ import annotations

import re

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import MODEL_EMBED, SEMANTIC_BREAKPOINT_PERCENTILE


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    """Naive but reliable for SEC-filing prose. Avoids spaCy as a heavy dep."""
    # Pre-clean numbered headings that confuse the regex (e.g. "1.\nRisk Factors")
    text = re.sub(r"(\d+)\.\s*\n", r"\1. ", text)
    parts = _SENTENCE_RE.split(text)
    # Strip empties and microscopic fragments.
    return [p.strip() for p in parts if len(p.strip()) > 4]


def _semantic_chunks(sentences: list[str], embeddings, percentile: int) -> list[str]:
    """Group sentences into chunks where each break is a semantic discontinuity."""
    if len(sentences) <= 1:
        return ["\n".join(sentences)] if sentences else []

    # Embed all sentences in one batch (fast, cheap).
    vecs = embeddings.embed_documents(sentences)
    vecs = np.array(vecs)

    # Cosine distance between consecutive sentences.
    distances = []
    for i in range(len(sentences) - 1):
        a, b = vecs[i], vecs[i + 1]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        distances.append(1.0 - sim)

    # Break wherever distance exceeds the chosen percentile.
    threshold = np.percentile(distances, percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    chunks, start = [], 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start : bp + 1]))
        start = bp + 1
    chunks.append(" ".join(sentences[start:]))
    return [c for c in chunks if c.strip()]


def chunk_pages(pages: list[dict]) -> list[Document]:
    """Turn page records into chunked LangChain Documents with citation metadata.

    Each output Document has metadata = {"source": str, "page": int}.
    """
    embeddings = OpenAIEmbeddings(model=MODEL_EMBED)
    docs: list[Document] = []

    for p in pages:
        sentences = _split_sentences(p["text"])
        if not sentences:
            continue

        # Pages with very few sentences: keep as one chunk to save embedding calls.
        if len(sentences) < 4:
            chunks = [" ".join(sentences)]
        else:
            chunks = _semantic_chunks(sentences, embeddings, SEMANTIC_BREAKPOINT_PERCENTILE)

        for c in chunks:
            docs.append(Document(
                page_content=c,
                metadata={"source": p["source"], "page": p["page"]},
            ))

    return docs
