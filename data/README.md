# `data/` directory

Runtime artifacts. Everything in here except this README is gitignored.

## Layout

```
data/
├── pdfs/                # source PDFs (auto-downloaded for samples)
│   ├── Apple_10K.pdf
│   ├── Tesla_10K.pdf
│   └── (any PDF you upload through the UI)
└── chroma_db/           # generated vector index
```

## Bundled samples

When you run `python ingest.py`, the script auto-downloads Apple's and Tesla's most recent 10-K filings if they aren't already present. These are public SEC filings — no auth needed.

If the download fails (firewall, network, etc.), drop your own 10-K PDF into `data/pdfs/` and re-run `python ingest.py` — it picks up everything in that folder.

## Adding more documents

Either:
- Drop additional PDFs into `data/pdfs/` and run `python ingest.py` (full rebuild).
- Use the upload widget in the Gradio UI — appends to the existing index without rebuilding.
