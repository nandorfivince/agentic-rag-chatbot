"""search_documents tool (RAG) -- a RAG subgraph-ot hivja.

Ez az egyetlen RAG tool. A tenyleges retrieve/rerank/format munkat a RAG
subgraph (graph/rag_subgraph.py) vegzi, ez a tool csak meghivja. Ezaltal a
subgraph moduláris maradnak -- lusta hivatkozasu import egy-import ciklust
kerul.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from tools import ToolContext


def build_search_documents_tool(context: ToolContext) -> BaseTool:
    @tool
    def search_documents(query: str) -> str:
        """Szemantikus + kulcsszavas (hibrid) kereses a dokumentumok teljes szovegeben.

        Hasznald amikor konkret informaciot keresel a dokumentumokban:
        szerzodesi klauzulak, datumok, tetelek, kotbér, stb. A kereses
        hibrid: vektor-alapu szemantikai kozelseg + BM25 kulcsszo-overlap.

        Args:
            query: Keresesi kifejezes magyarul (pl. "szallitasi hatarido").

        Visszater: JSON tömb top-5 relevancia-rangsorolt talalattal, forrassal
        es relevancia-scorerrel.
        """
        if context.trace_hook:
            context.trace_hook("search_documents", {"query": query})

        if context.store is None:
            return json.dumps({"error": "A vektortar nincs inicializalva."},
                              ensure_ascii=False)

        # Lusta import: a RAG subgraph-ot a graph reteg tartalmazza
        from graph.rag_subgraph import run_rag_subgraph

        result = run_rag_subgraph(query=query, store=context.store, top_k=5)
        return json.dumps(result, ensure_ascii=False, indent=2)

    return search_documents
