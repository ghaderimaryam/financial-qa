"""Build and load the Chroma vector store."""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import CHROMA_PATH, COLLECTION_NAME, MODEL_EMBED


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=MODEL_EMBED)


def build_vectorstore(documents: list[Document], rebuild: bool = True) -> Chroma:
    if rebuild and CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print("Cleared old vector store")

    vs = Chroma.from_documents(
        documents=documents,
        embedding=_embeddings(),
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )
    print(f"Vector store ready with {vs._collection.count()} chunks")
    return vs


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=_embeddings(),
        collection_name=COLLECTION_NAME,
    )
