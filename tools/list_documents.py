"""list_documents tool -- feltoltott dokumentumok listazasa (non-RAG)."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from ingest.extraction import classify_heuristic
from tools import ToolContext  # forward ref ok, import at call time


def build_list_documents_tool(context: ToolContext) -> BaseTool:
    """Epit egy list_documents tool-t a context alapjan."""

    @tool
    def list_documents() -> str:
        """Feltoltott es feldolgozott dokumentumok listazasa tipusossan.

        Hasznald ezt a tool-t eloszor, mielott specifikus tool-okat hivnal,
        hogy tudd milyen fajlok allnak rendelkezesre es milyen tipusuak.

        Visszater: JSON objektum {"files": [{"filename", "doc_type", "size_chars"}]}.
        doc_type lehet: "szamla", "szerzodes", "szallitolevel", "megrendeles", "egyeb".
        """
        if context.trace_hook:
            context.trace_hook("list_documents", {})

        files = []
        for name, doc in context.documents.items():
            files.append({
                "filename": name,
                "doc_type": classify_heuristic(doc.full_text),
                "size_chars": len(doc.full_text),
            })

        return json.dumps(
            {"files": files, "total": len(files)},
            ensure_ascii=False,
            indent=2,
        )

    return list_documents
