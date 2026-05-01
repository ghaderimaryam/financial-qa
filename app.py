"""Launch the Gradio Q&A interface.

Usage:
    python app.py

Prerequisite: run `python ingest.py` once to build the vector index.
"""
from __future__ import annotations

from src.ui import build_demo


def main() -> None:
    demo, css, theme = build_demo()
    demo.launch(theme=theme, css=css, inbrowser=True)


if __name__ == "__main__":
    main()
