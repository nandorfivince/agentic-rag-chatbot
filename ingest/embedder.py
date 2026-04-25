"""Embedding: sentence-transformers lokalis, multilingual modell.

Alapertelmezett: `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, 118 MB).
Magyar szovegre is jo, ingyenes, CPU-n is gyorsan fut.

A modell singleton-kent betoltodik az elso hivasnal, utana cache-elve marad.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Union

from config import settings


@lru_cache(maxsize=1)
def _load_model():
    """A sentence-transformers modell lusta betoltese (egyszer az elso hivasra)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


def embed(text: str) -> list[float]:
    """Egyetlen szoveg embed-elese."""
    model = _load_model()
    vector = model.encode(text, show_progress_bar=False)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding -- gyorsabb, mint egyesevel."""
    if not texts:
        return []
    model = _load_model()
    matrix = model.encode(texts, show_progress_bar=False, batch_size=32)
    return matrix.tolist()


def embedding_dim() -> int:
    """A modell dimenzioja -- ChromaDB-nek nem kell explicit, csak hibakeresesre."""
    return _load_model().get_sentence_embedding_dimension()
