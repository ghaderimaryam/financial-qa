"""Gradio UI for the Financial Q&A.

Layout: question box on the left, answer + citations + faithfulness panel on the right.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr

from src import config
from src.pipeline import ask
from src.vector_store import load_vectorstore


CUSTOM_CSS = """
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem 3rem !important;
}

#hero {
    background: linear-gradient(135deg, rgba(15,23,42,0.04) 0%, rgba(99,102,241,0.06) 100%);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 1.5rem 1.85rem;
    margin-bottom: 1.5rem;
}
#hero h1 {
    font-size: 1.5rem; font-weight: 700; margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #0f172a, #4f46e5 80%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: inline-block;
}
#hero p {
    color: var(--body-text-color-subdued); font-size: 0.93rem;
    margin: 0; max-width: 760px; line-height: 1.55;
}

.section-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--body-text-color-subdued);
    margin: 1.4rem 0 0.65rem;
}

/* ── Faithfulness banner ─────────────────────────────── */
.faith-banner {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 0.95rem 1.15rem; border-radius: 10px;
    border: 1px solid;
    margin-bottom: 0.6rem;
}
.faith-banner.green  { background: rgba(16,185,129,0.06); border-color: rgba(16,185,129,0.4); }
.faith-banner.yellow { background: rgba(245,158,11,0.06); border-color: rgba(245,158,11,0.45); }
.faith-banner.red    { background: rgba(220,38,38,0.06); border-color: rgba(220,38,38,0.5); }

.faith-icon { font-size: 1.4rem; }
.faith-text { flex: 1; }
.faith-title { font-weight: 600; font-size: 0.95rem; color: var(--body-text-color); }
.faith-summary { font-size: 0.85rem; color: var(--body-text-color-subdued); margin-top: 0.15rem; }
.faith-score {
    font-size: 1.2rem; font-weight: 700; font-variant-numeric: tabular-nums;
}
.faith-banner.green  .faith-score { color: #059669; }
.faith-banner.yellow .faith-score { color: #d97706; }
.faith-banner.red    .faith-score { color: #b91c1c; }

/* ── Answer box ─────────────────────────────────────── */
.answer-box {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    line-height: 1.65;
    font-size: 0.95rem;
}
.answer-box .citation {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    color: #4338ca;
    padding: 0.05rem 0.45rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0 0.1rem;
    font-variant-numeric: tabular-nums;
}

/* ── Claim breakdown ────────────────────────────────── */
.claim-list { display: flex; flex-direction: column; gap: 0.5rem; }
.claim {
    display: flex; gap: 0.7rem; align-items: flex-start;
    padding: 0.65rem 0.85rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    font-size: 0.85rem;
    background: var(--background-fill-secondary);
}
.claim-verdict {
    flex: 0 0 auto;
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
}
.claim-verdict.yes     { background: rgba(16,185,129,0.15); color: #059669; }
.claim-verdict.partial { background: rgba(245,158,11,0.15); color: #d97706; }
.claim-verdict.no      { background: rgba(220,38,38,0.15); color: #b91c1c; }
.claim-text { flex: 1; }
.claim-reason { color: var(--body-text-color-subdued); font-size: 0.78rem; margin-top: 0.2rem; font-style: italic; }

footer { display: none !important; }
"""


HERO_HTML = """
<div id="hero">
    <h1>📑 Financial Document Q&A</h1>
    <p>
        A RAG system for SEC 10-K filings — built on Apple, Tesla, and NVIDIA.
        Ask any question, and the system retrieves relevant passages from the filings,
        generates an answer with <strong>inline page-level citations</strong>, and runs a
        <strong>two-stage faithfulness check</strong> that extracts each factual claim and
        verifies it against the source. Unsupported answers are flagged or blocked before
        they reach you.
    </p>
    <p style="margin-top: 0.65rem; font-size: 0.82rem; color: var(--body-text-color-subdued);">
        Try a sample question below, or upload your own filing to extend the index.
    </p>
</div>
"""


SAMPLE_QUESTIONS = [
    "What were Apple's total net sales in fiscal 2023?",
    "How does Tesla describe its main competitive risks?",
    "What does NVIDIA say about its data center business segment?",
    "Compare gross margins of Apple and NVIDIA.",
    "What are NVIDIA's main competitive risks according to the 10-K?",
    "Compare R&D spending between Apple and Tesla.",
    "How does NVIDIA discuss AI demand in its risk factors?",
    "What does Apple say about its services segment growth?",
]


# ─── Helpers ──────────────────────────────────────────────────────────────
_CITATION_RE = re.compile(r"\[([^\[\]]+?)\s+p\.(\d+)\]")


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_answer(text: str) -> str:
    """Wrap citation tags in a span so CSS can style them as pills."""
    safe = _esc(text)
    safe = _CITATION_RE.sub(
        lambda m: f'<span class="citation">{_esc(m.group(1))} p.{m.group(2)}</span>',
        safe,
    )
    safe = safe.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f'<div class="answer-box"><p>{safe}</p></div>'


def _render_faithfulness(faith: dict) -> str:
    verdict = faith["verdict"]
    pct = f"{faith['score']:.0%}"
    title_map = {
        "green": "Faithful",
        "yellow": "Possible inconsistency",
        "red": "Faithfulness check failed",
    }
    icon_map = {"green": "✅", "yellow": "⚠️", "red": "❌"}
    return f"""
    <div class="faith-banner {verdict}">
        <div class="faith-icon">{icon_map[verdict]}</div>
        <div class="faith-text">
            <div class="faith-title">{title_map[verdict]}</div>
            <div class="faith-summary">{_esc(faith['summary'])}</div>
        </div>
        <div class="faith-score">{pct}</div>
    </div>
    """


def _render_claims(claims: list[dict]) -> str:
    if not claims:
        return '<p style="color: var(--body-text-color-subdued); font-size: 0.85rem;">No verifiable claims (e.g. system declined to answer).</p>'
    items = []
    for c in claims:
        items.append(f"""
        <div class="claim">
            <span class="claim-verdict {c['verdict']}">{c['verdict']}</span>
            <div class="claim-text">
                {_esc(c['claim'])}
                <div class="claim-reason">{_esc(c['reason'])}</div>
            </div>
        </div>""")
    return f'<div class="claim-list">{"".join(items)}</div>'


def _render_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    pills = " ".join(
        f'<span class="citation">{_esc(c["source"])} p.{c["page"]}</span>'
        for c in citations
    )
    return f'<div style="margin-top: 0.5rem; font-size: 0.85rem;"><strong>Sources used:</strong> {pills}</div>'


# ─── Build app ────────────────────────────────────────────────────────────
def build_demo():
    config.validate()
    if not config.CHROMA_PATH.exists():
        raise SystemExit(
            f"❌ No vector store at {config.CHROMA_PATH}. "
            "Run `python ingest.py` first."
        )
    vectorstore = load_vectorstore()
    print(f"✅ Vector store loaded ({vectorstore._collection.count()} chunks)")

    theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_md,
    )

    with gr.Blocks(title="Financial Document Q&A") as demo:
        gr.HTML(HERO_HTML)

        with gr.Row():
            # ── Left column: question + upload ────────────────
            with gr.Column(scale=2):
                gr.HTML('<div class="section-label">Your question</div>')
                question = gr.Textbox(
                    placeholder="e.g. What were Apple's total net sales in fiscal 2023?",
                    show_label=False, lines=3,
                )
                ask_btn = gr.Button("Ask", variant="primary")

                gr.HTML('<div class="section-label">Or try a sample</div>')
                sample_dd = gr.Dropdown(
                    choices=SAMPLE_QUESTIONS,
                    label="Sample questions",
                    show_label=False,
                )

                gr.HTML('<div class="section-label">Add your own filing</div>')
                upload = gr.File(label="Upload a PDF", file_types=[".pdf"], height=110)
                upload_status = gr.Markdown()

            # ── Right column: answer + faithfulness ───────────
            with gr.Column(scale=3):
                gr.HTML('<div class="section-label">Faithfulness check</div>')
                faith_html = gr.HTML(value="""
                    <div class="faith-banner" style="border-color: var(--border-color-primary);">
                        <div class="faith-icon">⏳</div>
                        <div class="faith-text">
                            <div class="faith-title">Awaiting question</div>
                            <div class="faith-summary">Ask something to see the faithfulness score.</div>
                        </div>
                    </div>
                """)

                gr.HTML('<div class="section-label">Answer</div>')
                answer_html = gr.HTML(value='<div class="answer-box" style="color: var(--body-text-color-subdued);">Your answer will appear here, with citations linking back to the source filings.</div>')
                citations_html = gr.HTML()

                with gr.Accordion("🔍 Per-claim breakdown", open=False):
                    claims_html = gr.HTML()

        # ─── Wiring ───────────────────────────────────────────
        def on_ask(q: str):
            q = (q or "").strip()
            if not q:
                return (
                    gr.update(),  # faith
                    '<div class="answer-box" style="color:#dc2626;">Please enter a question.</div>',
                    "",
                    "",
                )
            try:
                result = ask(vectorstore, q)
            except Exception as e:
                err = f'<div class="answer-box" style="color:#dc2626;">Error: {_esc(repr(e))}</div>'
                return gr.update(), err, "", ""
            return (
                _render_faithfulness(result["faithfulness"]),
                _render_answer(result["answer"]),
                _render_citations(result["citations"]),
                _render_claims(result["faithfulness"]["claims"]),
            )

        def on_sample_pick(s: str):
            return s

        def on_upload(file_obj) -> tuple[str, str]:
            if file_obj is None:
                return "", ""
            try:
                src_path = Path(file_obj.name)
                target = config.PDF_DIR / src_path.name
                config.PDF_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, target)
                # Add to existing index — same Python that's running the UI.
                proc = subprocess.run(
                    [sys.executable, "ingest.py", "--add", str(target)],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    return (
                        f"❌ Ingest failed: `{_esc(proc.stderr.strip()[-200:])}`",
                        "",
                    )
                # Force reload of in-memory vectorstore so new chunks are queryable.
                nonlocal vectorstore
                vectorstore = load_vectorstore()
                return f"✅ `{src_path.name}` added to the index.", ""
            except Exception as e:
                return f"❌ Upload failed: {_esc(repr(e))}", ""

        ask_btn.click(on_ask, [question], [faith_html, answer_html, citations_html, claims_html])
        question.submit(on_ask, [question], [faith_html, answer_html, citations_html, claims_html])
        sample_dd.change(on_sample_pick, [sample_dd], [question])
        upload.change(on_upload, [upload], [upload_status, upload_status])

    return demo, CUSTOM_CSS, theme
