"""Kozponti konfiguracio -- env valtozokat olvas, default ertekekkel.

Ezt az osszes modul importalhatja (llm, ingest, graph, tools), igy egyetlen
helyrol allithato a provider/modell/chunking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = _env("LLM_PROVIDER", "ollama")  # ollama | dummy
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.1:8b")

    # Embedding
    embedding_model: str = _env(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ChromaDB
    chroma_path: str = _env("CHROMA_PATH", "./chroma_db")
    chroma_collection: str = _env("CHROMA_COLLECTION", "documents")

    # Chunking
    chunk_size: int = _env_int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 200)

    # Agentic loop
    max_iterations: int = _env_int("MAX_ITERATIONS", 8)
    max_validator_retries: int = _env_int("MAX_VALIDATOR_RETRIES", 2)

    # Streamlit
    streamlit_port: int = _env_int("STREAMLIT_PORT", 8501)


settings = Settings()
