"""Hibrid vektortar: ChromaDB (vector) + BM25Okapi (sparse) RRF fusionnal.

Miert hibrid? A vektor-kereses a szemantikai kozelseget fogja meg
("targy" ~ "teszt_anyag"), a BM25 a pontos szot ("ISIN: HU0000123456").
Egyutt robusztusabb -- egyik sem hibaztok rosszul.

Reciprocal Rank Fusion (RRF): az egyik legegyszerubb, de hatekony hibrid
fusion. Sulytalan, hangolas nelkul mukodik. score = 1 / (k + rank).
"""

from __future__ import annotations

import uuid
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

from config import settings


def _tokenize(text: str) -> list[str]:
    """Egyszerű tokenizer BM25-hez: kisbetus, nem-alfanumerikus splitnel."""
    import re
    return [t for t in re.split(r"\W+", text.lower()) if t]


class HybridStore:
    """ChromaDB + BM25 hibrid vektortar.

    Hasznalat:
        store = HybridStore()
        store.add_chunks([{"text": "...", "metadata": {"source": "a.pdf"}}])
        results = store.search("mennyi a vegosszeg?", top_k=5)

    Perzisztencia: Chroma automatikusan kezeli (persist_path). A BM25 index
    memoria-bazisu, minden add_chunks utan ujraepul -- feltoltott doksik
    mennyisegen (10-100) ez sosem szuk keresztmetszet.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.client = chromadb.PersistentClient(path=persist_path or settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 in-memory cache
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: list[dict] = []  # [{text, metadata, id}, ...]
        self._rebuild_bm25_from_collection()

    def _rebuild_bm25_from_collection(self) -> None:
        """Chroma-bol kiolvassuk a mar tarolt chunk-okat, BM25 indexet epitunk."""
        try:
            data = self.collection.get(include=["documents", "metadatas"])
        except Exception:
            # Ures kollekcio
            self._bm25 = None
            self._bm25_chunks = []
            return

        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        ids = data.get("ids") or []

        self._bm25_chunks = [
            {"text": d, "metadata": m or {}, "id": i}
            for d, m, i in zip(docs, metas, ids)
        ]
        if self._bm25_chunks:
            corpus = [_tokenize(c["text"]) for c in self._bm25_chunks]
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Chunk-ok betoltese Chroma-ba + BM25 index ujraepitese.

        chunks: [{"text": str, "metadata": dict}, ...]
        embeddings: [[float], ...] chunk-onkenti vektor (ugyanakkora hossz)
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks es embeddings hossza nem egyezik")

        ids = [f"chunk_{uuid.uuid4().hex}" for _ in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._rebuild_bm25_from_collection()

    def _search_vector(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Chroma vektor-kereses."""
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        return [
            {
                "text": d,
                "metadata": m or {},
                "id": i,
                "distance": dist,
                "score_vector": 1.0 - dist,  # cosine distance -> similarity
            }
            for d, m, i, dist in zip(docs, metas, ids, dists)
        ]

    def _search_bm25(self, query: str, top_k: int) -> list[dict]:
        """BM25 kereses -- pontos kulcsszo-overlap."""
        if self._bm25 is None or not self._bm25_chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Top-k index
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            {
                **self._bm25_chunks[idx],
                "score_bm25": float(score),
            }
            for idx, score in ranked
            if score > 0
        ]

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Hibrid keresés: vektor + BM25 + Reciprocal Rank Fusion.

        Visszater: [{"text", "metadata", "id", "score"}, ...] top_k-ig
        """
        vector_hits = self._search_vector(query_embedding, top_k=top_k * 2)
        bm25_hits = self._search_bm25(query, top_k=top_k * 2)

        # RRF fusion -- score = 1 / (k + rank), k = 60 (standard)
        k_rrf = 60
        rrf: dict[str, dict] = {}

        for rank, hit in enumerate(vector_hits):
            hid = hit["id"]
            rrf.setdefault(hid, {"hit": hit, "score": 0.0})
            rrf[hid]["score"] += 1.0 / (k_rrf + rank + 1)

        for rank, hit in enumerate(bm25_hits):
            hid = hit["id"]
            rrf.setdefault(hid, {"hit": hit, "score": 0.0})
            rrf[hid]["score"] += 1.0 / (k_rrf + rank + 1)

        # Top-k RRF score szerint
        ranked = sorted(rrf.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        return [
            {
                "text": r["hit"]["text"],
                "metadata": r["hit"]["metadata"],
                "id": r["hit"]["id"],
                "score": r["score"],
            }
            for r in ranked
        ]

    def list_sources(self) -> list[str]:
        """Egyedi fajlnevek a kollekcioban (a `source` metadata alapjan)."""
        sources: set[str] = set()
        for chunk in self._bm25_chunks:
            src = chunk["metadata"].get("source")
            if src:
                sources.add(src)
        return sorted(sources)

    def count(self) -> int:
        """Chunk-ok szama."""
        return self.collection.count()

    def clear(self) -> None:
        """Teljes kollekcio torlese -- tesztekhez."""
        name = self.collection.name
        self.client.delete_collection(name=name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_chunks = []
