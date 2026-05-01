"""End-to-end Q&A pipeline. Used by both the Gradio UI and FastAPI service.

ask(question) → retrieve → generate answer → evaluate faithfulness → return result.

The shape of the returned dict is what the API serializes and what the UI renders.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from langchain_chroma import Chroma

from src.faithfulness import FaithfulnessReport, evaluate
from src.qa_chain import answer as generate_answer
from src.retrieval import format_context, retrieve, unique_citations


def ask(vectorstore: Chroma, question: str, top_k: Optional[int] = None) -> dict:
    """Run the full Q&A pipeline and return a structured response."""
    docs = retrieve(vectorstore, question, k=top_k) if top_k else retrieve(vectorstore, question)
    context = format_context(docs)
    citations = unique_citations(docs)

    raw_answer = generate_answer(question, context)
    report: FaithfulnessReport = evaluate(raw_answer, context)

    # If the score is below the block threshold, replace the answer with a
    # transparent refusal that names the reason.
    if report.should_block:
        served_answer = (
            "⚠️  This answer was withheld because the faithfulness check found "
            f"too many unsupported claims (score {report.score:.0%}). "
            "Try rephrasing your question, or ask about a topic clearly covered "
            "in the filings."
        )
    else:
        served_answer = raw_answer

    return {
        "question": question,
        "answer": served_answer,
        "raw_answer": raw_answer,                    # what the LLM produced before guardrail
        "blocked": report.should_block,
        "citations": citations,
        "faithfulness": {
            "score": report.score,
            "verdict": report.verdict,
            "summary": report.summary,
            "claims": [asdict(c) for c in report.claims],
        },
    }
