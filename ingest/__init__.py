"""Dokumentum betoltes + chunking + embedding."""

from ingest.chunker import chunk_document, chunk_text
from ingest.embedder import embed, embed_batch, embedding_dim
from ingest.pdf_loader import Document, Page, load_bytes, load_docx, load_image, load_pdf

__all__ = [
    "Document",
    "Page",
    "load_bytes",
    "load_pdf",
    "load_docx",
    "load_image",
    "chunk_text",
    "chunk_document",
    "embed",
    "embed_batch",
    "embedding_dim",
]
