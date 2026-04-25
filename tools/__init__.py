"""LangChain tool-ok: 5 tool, kozul 1 RAG, 4 non-RAG.

A tool-ok state-fuggok (feltoltott dokumentumok, ChromaDB peldany) ezert
factory mintaval epitjuk: `build_tools(context)` -> list[BaseTool]. A context
az app.py session-allapotban el.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_core.tools import BaseTool

from ingest.pdf_loader import Document
from store.vector_store import HybridStore


@dataclass
class ToolContext:
    """A tool-ok kozos allapota -- app.py session state-bol tapalja."""
    store: Optional[HybridStore] = None
    documents: dict[str, Document] = field(default_factory=dict)
    extractions: dict[str, dict] = field(default_factory=dict)
    trace_hook: Optional[Callable[[str, dict], None]] = None


def build_tools(context: ToolContext) -> list[BaseTool]:
    """5 LangChain tool epitese a megadott context-szel."""
    from tools.list_documents import build_list_documents_tool
    from tools.get_extraction import build_get_extraction_tool
    from tools.rag_search import build_search_documents_tool
    from tools.compare_documents import build_compare_documents_tool
    from tools.validate_document import build_validate_document_tool

    return [
        build_list_documents_tool(context),
        build_get_extraction_tool(context),
        build_search_documents_tool(context),
        build_compare_documents_tool(context),
        build_validate_document_tool(context),
    ]


__all__ = ["ToolContext", "build_tools"]
