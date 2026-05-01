"""Faithfulness guardrail.

Two-stage check:
1. CLAIM EXTRACTION: ask the LLM to break the answer into atomic claims.
2. PER-CLAIM ENTAILMENT: for each claim, ask "is this claim supported by the
   retrieved context? yes/partial/no" with temperature=0 for determinism.

Score = (yes count + 0.5 * partial count) / total claims. Higher = more faithful.

Why two stages instead of "is the whole answer faithful?":
- A single yes/no over a long answer hides per-sentence hallucinations.
- Atomic claims let the UI highlight WHICH claim is unsupported, not just that
  something is wrong overall.
- Decoupling extraction from judgment makes the score more stable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config import (
    FAITHFULNESS_BLOCK_BELOW,
    FAITHFULNESS_WARN_BELOW,
    MODEL_JUDGE,
)

Verdict = Literal["yes", "partial", "no"]


@dataclass
class ClaimCheck:
    claim: str
    verdict: Verdict
    reason: str


@dataclass
class FaithfulnessReport:
    score: float                     # 0.0 - 1.0
    verdict: Literal["green", "yellow", "red"]
    should_block: bool
    claims: list[ClaimCheck]
    summary: str                     # human-readable one-liner


# ─── Stage 1: claim extraction ────────────────────────────────────────────
_EXTRACT_PROMPT = """Break the following answer into a list of atomic factual claims.

An atomic claim is a single, self-contained statement of fact (e.g. "Apple's R&D was $30.0B in fiscal 2023" is one claim; "Apple's R&D was $30.0B and rose 14% YoY" is two).

Ignore citation tags like [Apple_10K p.47] — they are not claims.
Ignore meta-statements like "according to the filings" — they are not claims.
Ignore refusal statements like "the filings do not contain this information" — return an empty list.

Return ONLY a JSON array of strings. Example: ["Claim one.", "Claim two."]

Answer:
{answer}"""


def _extract_claims(answer: str, judge: ChatOpenAI) -> list[str]:
    response = judge.invoke([HumanMessage(content=_EXTRACT_PROMPT.format(answer=answer))])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        claims = json.loads(text)
        return [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    except json.JSONDecodeError:
        return []


# ─── Stage 2: per-claim entailment ────────────────────────────────────────
_JUDGE_PROMPT = """You are checking whether a CLAIM is supported by a CONTEXT extracted from SEC filings.

CLAIM:
{claim}

CONTEXT:
{context}

Rules:
- "yes" — the claim is fully and explicitly supported by the context.
- "partial" — part of the claim is supported, but a number, qualifier, or detail is missing or different.
- "no" — the claim contradicts the context, OR the context does not mention the topic of the claim at all.

Return ONLY a JSON object: {{"verdict": "yes|partial|no", "reason": "one short sentence"}}"""


def _judge_claim(claim: str, context: str, judge: ChatOpenAI) -> ClaimCheck:
    prompt = _JUDGE_PROMPT.format(claim=claim, context=context[:6000])
    response = judge.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        obj = json.loads(text)
        verdict = obj.get("verdict", "no").lower()
        if verdict not in ("yes", "partial", "no"):
            verdict = "no"
        return ClaimCheck(claim=claim, verdict=verdict, reason=obj.get("reason", ""))
    except json.JSONDecodeError:
        return ClaimCheck(claim=claim, verdict="no", reason="Judge response unparseable")


# ─── Public API ───────────────────────────────────────────────────────────
def evaluate(answer: str, context: str) -> FaithfulnessReport:
    """Run the full two-stage check and return a report."""
    judge = ChatOpenAI(model=MODEL_JUDGE, temperature=0)

    claims = _extract_claims(answer, judge)
    if not claims:
        # No verifiable claims (e.g. polite refusal) — treat as faithful by default.
        return FaithfulnessReport(
            score=1.0,
            verdict="green",
            should_block=False,
            claims=[],
            summary="No factual claims to verify.",
        )

    checks = [_judge_claim(c, context, judge) for c in claims]
    yes = sum(1 for c in checks if c.verdict == "yes")
    partial = sum(1 for c in checks if c.verdict == "partial")
    score = (yes + 0.5 * partial) / len(checks)

    if score < FAITHFULNESS_BLOCK_BELOW:
        verdict, should_block = "red", True
    elif score < FAITHFULNESS_WARN_BELOW:
        verdict, should_block = "yellow", False
    else:
        verdict, should_block = "green", False

    summary = f"{yes}/{len(checks)} claims fully supported"
    if partial:
        summary += f", {partial} partially"
    no_count = len(checks) - yes - partial
    if no_count:
        summary += f", {no_count} unsupported"

    return FaithfulnessReport(
        score=score,
        verdict=verdict,  # type: ignore[arg-type]
        should_block=should_block,
        claims=checks,
        summary=summary,
    )
