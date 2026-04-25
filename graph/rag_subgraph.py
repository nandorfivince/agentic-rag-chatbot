"""RAG subgraph -- retrieve -> rerank -> format.

Dedikalt modularis LangGraph subgraph amit a `search_documents` tool invokal.
A plan szerint ez nem szamit a main-graph 5 node-jaba, ezert kulon graphot
epitunk es compile-olunk.

Architektura:

    retrieve   -- HybridStore.search() + embed(query) = top_2k raw hits
        |
    rerank     -- kulcsszo-overlap boost + top_k levagas
        |
    format     -- forras-cimkezett output: [{rank, source, page, score, text}]

A store-t closure-kent kapja a graph-epitő (build_rag_graph(store)), igy nem
kell a state-ben nyilvantartani -- ez egyben a checkpoint kompatibilitast is
biztositja, ha kesobb perzisztens memory kellene.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict, total=False):
    query: str
    top_k: int
    raw_hits: list[dict]
    reranked_hits: list[dict]
    output: list[dict]


# Per-store compiled graph cache (az app.py egyetlen store-t hasznal
# session-enkent, de az eval-teszt tobbet is futtathat).
_RAG_GRAPH_CACHE: dict[int, Any] = {}


def _tokenize_for_boost(text: str) -> set[str]:
    """Csak > 3 karakteres, alfanumerikus tokenek -- a rövideket kihagyjuk (az, a, is)."""
    return {t for t in re.split(r"\W+", text.lower()) if len(t) > 3}


def build_rag_graph(store):
    """RAG subgraph epitese a store-ral mint closure.

    A `store` egy HybridStore peldany -- a retrieve node erre hivatkozik.
    """
    # Lazy import: a store-modult csak igeny eseten toltjuk, igy a graph-modul
    # tesztelheto a store-function nelkul is.
    from ingest.embedder import embed

    def retrieve_node(state: RAGState) -> dict:
        query = state.get("query", "")
        top_k = state.get("top_k", 5)
        if not query.strip():
            return {"raw_hits": []}

        # Az embed egy hosszabb muvelet (sentence-transformer forward pass),
        # ezert ez a RAG subgraph legkoltsegesebb lepese. A `embed()` a
        # sentence-transformers modellt singleton-kent cache-eli (lru_cache),
        # igy az elso hivas utan gyors.
        query_embedding = embed(query)
        hits = store.search(query, query_embedding, top_k=top_k * 2)
        return {"raw_hits": hits}

    def rerank_node(state: RAGState) -> dict:
        hits = state.get("raw_hits", [])
        top_k = state.get("top_k", 5)
        query_tokens = _tokenize_for_boost(state.get("query", ""))

        for hit in hits:
            text_lower = (hit.get("text") or "").lower()
            overlap = sum(1 for t in query_tokens if t in text_lower)
            # Minden overlap +0.01 boost a hibrid RRF scorera
            hit["score_reranked"] = hit.get("score", 0.0) + 0.01 * overlap

        ranked = sorted(
            hits, key=lambda h: h.get("score_reranked", 0.0), reverse=True
        )[:top_k]
        return {"reranked_hits": ranked}

    def format_node(state: RAGState) -> dict:
        out = []
        for rank, hit in enumerate(state.get("reranked_hits", []), start=1):
            metadata = hit.get("metadata", {}) or {}
            text = hit.get("text", "") or ""
            out.append({
                "rank": rank,
                "source": metadata.get("source", "?"),
                "page": metadata.get("page", 1),
                "chunk_index": metadata.get("chunk_index"),
                "score": round(
                    hit.get("score_reranked", hit.get("score", 0.0)), 4
                ),
                "text": text if len(text) <= 500 else text[:500] + "...",
            })
        return {"output": out}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("format", format_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "format")
    graph.add_edge("format", END)
    return graph.compile()


def run_rag_subgraph(query: str, store, top_k: int = 5) -> dict:
    """A RAG subgraph futtatasa -- a `search_documents` tool hivja.

    Visszater: {"query", "hits": [{rank, source, page, score, text}], "count"}.
    """
    key = id(store)
    if key not in _RAG_GRAPH_CACHE:
        _RAG_GRAPH_CACHE[key] = build_rag_graph(store)
    compiled = _RAG_GRAPH_CACHE[key]

    initial: RAGState = {"query": query, "top_k": top_k}
    result = compiled.invoke(initial)

    return {
        "query": query,
        "hits": result.get("output", []),
        "count": len(result.get("output", [])),
    }
