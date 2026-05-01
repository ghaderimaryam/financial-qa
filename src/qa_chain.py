"""Generates an answer with inline page-level citations.

The prompt is the load-bearing part of this module: it forces the model to cite
every claim with a `[Source p.N]` tag and to refuse politely if the context does
not contain enough information to answer.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import MODEL_GEN

_SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC filings.

You will receive a user question and several context passages, each prefixed with a citation tag like [Apple_10K p.47].

Strict rules:
1. Use ONLY the information in the provided context. Do not use prior knowledge.
2. After every factual claim, append the matching citation tag exactly as it appears, e.g. "Apple's R&D was $30.0B [Apple_10K p.47]."
3. If a claim draws on multiple passages, cite all of them: "[Apple_10K p.47][Apple_10K p.48]".
4. 4. If a specific number or fact is missing from the context, say so for THAT detail — but still answer the parts of the question you CAN answer from the context. Only refuse the entire question if the context is genuinely silent on the topic.
5. Be precise with numbers — quote them exactly as they appear in the context, with units (e.g. "$30.0 billion" not "$30B").
6. Keep the answer focused. Two to four short paragraphs. No bullet points unless the question explicitly asks to compare items.

Context:
{context}"""


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{question}"),
])


def answer(question: str, context: str, temperature: float = 0.1) -> str:
    """Run the QA chain. Returns the answer string with inline citations."""
    llm = ChatOpenAI(model=MODEL_GEN, temperature=temperature)
    chain = _PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
