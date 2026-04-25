"""validate_document tool -- matematikai + logikai validacio egy dokumentumon.

A `utils.validation.validate_all()`-t hivja: netto+AFA=brutto, tetel-szintu
matek, datum-sorrend, adoszam mod-11 CDV, plauzibilitas.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from tools import ToolContext
from tools.get_extraction import _get_or_extract
from utils.validation import validate_all


def build_validate_document_tool(context: ToolContext) -> BaseTool:
    @tool
    def validate_document(filename: str) -> str:
        """Matematikai es logikai validacio futtatasa egy dokumentumon.

        Szamlan ellenorzi:
        - netto + AFA = brutto
        - tetelek osszege = netto vegosszeg
        - tetel-szintu mennyiseg x egysegar = tetel netto
        - magyar adoszam (XXXXXXXX-X-XX) mod-11 CDV helyesseg
        - datum-sorrend (kiallitas <= teljesites <= fizetesi hatarido)
        - plauzibilitas (negativ ertek, szokatlanul nagy osszeg, nem szabvanyos AFA kulcs)

        Args:
            filename: A dokumentum fajlneve.

        Visszater: JSON {"ok": bool, "errors": [...]}. Ha ok=true, nincs
        matematikai/logikai hiba a dokumentumban.
        """
        if context.trace_hook:
            context.trace_hook("validate_document", {"filename": filename})

        if filename not in context.documents:
            available = list(context.documents.keys())
            return json.dumps(
                {"error": f"Nem talalt dokumentum: '{filename}'",
                 "available": available},
                ensure_ascii=False,
            )

        data = _get_or_extract(context, filename)
        errors = validate_all(data)

        return json.dumps(
            {
                "ok": len(errors) == 0,
                "error_count": len(errors),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return validate_document
