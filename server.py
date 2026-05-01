"""FastAPI service exposing the Q&A pipeline.

Run:
    uvicorn server:app --reload

Endpoints:
    GET  /health           — basic liveness check
    POST /ask              — body: {"question": "..."} → answer + citations + faithfulness
    GET  /docs             — interactive Swagger UI (auto-generated)

The vector store is loaded once at startup and reused across requests.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.pipeline import ask
from src.vector_store import load_vectorstore


# ─── Request / response schemas ───────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Natural-language question about the filings.")
    top_k: int | None = Field(None, ge=1, le=20,
                              description="How many chunks to retrieve (default from config).")


class Citation(BaseModel):
    source: str
    page: int


class ClaimDetail(BaseModel):
    claim: str
    verdict: str
    reason: str


class FaithfulnessDetail(BaseModel):
    score: float
    verdict: str          # green | yellow | red
    summary: str
    claims: list[ClaimDetail]


class AskResponse(BaseModel):
    question: str
    answer: str
    raw_answer: str
    blocked: bool
    citations: list[Citation]
    faithfulness: FaithfulnessDetail


# ─── App lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.validate()
    if not config.CHROMA_PATH.exists():
        raise RuntimeError(
            f"No vector store at {config.CHROMA_PATH}. "
            "Run `python ingest.py` first."
        )
    app.state.vectorstore = load_vectorstore()
    print(f"✅ Vector store loaded ({app.state.vectorstore._collection.count()} chunks)")
    yield


app = FastAPI(
    title="Financial Document Q&A",
    description="Ask questions about SEC filings with page-level citations and a faithfulness guardrail.",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> dict:
    try:
        return ask(app.state.vectorstore, req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
