"""Retrieval primitives. Returns docs with page-level metadata intact."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import TOP_K


def retrieve(vectorstore: Chroma, question: str, k: int = TOP_K) -> list[Document]:
    """MMR retrieval — diversifies retrieved chunks across different pages.

    Plain similarity often returns 5 chunks from the same page (because they all
    score highly individually). For financial Q&A we want chunks from DIFFERENT
    pages — the table page, the discussion page, the YoY context. MMR penalizes
    near-duplicate chunks to give us that diversity.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 30, "lambda_mult": 0.5},
    )
    return retriever.invoke(question)


def format_context(docs: list[Document]) -> str:
    """Format retrieved docs with explicit citation tags.

    The tag format `[source p.N]` is what the LLM is taught to cite in answers,
    and what the UI later turns into clickable links.
    """
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        blocks.append(f"[{src} p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def unique_citations(docs: list[Document]) -> list[dict]:
    """De-duplicate retrieved docs into citation entries: [{source, page}, ...]."""
    seen = set()
    out: list[dict] = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            out.append({"source": key[0], "page": key[1]})
    return out
