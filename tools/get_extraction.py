"""get_extraction tool -- egy dokumentum kinyert strukturalt adatai (non-RAG).

A regex-alapu `ingest.extraction.extract_structured()`-et hivja. Az eredmenyt
cache-eli a context.extractions dict-ben, igy az ismetelt hivasok ingyenesek.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from ingest.extraction import extract_structured
from tools import ToolContext


def _get_or_extract(context: ToolContext, filename: str) -> dict:
    if filename in context.extractions:
        return context.extractions[filename]
    doc = context.documents.get(filename)
    if doc is None:
        return {}
    data = extract_structured(doc.full_text)
    context.extractions[filename] = data
    return data


def build_get_extraction_tool(context: ToolContext) -> BaseTool:
    @tool
    def get_extraction(filename: str) -> str:
        """Egy dokumentum kinyert strukturalt adatainak lekerdezese fajlnev alapjan.

        Szamla eseten: osszegek, datumok, felek, tetelek listaja.
        Szerzodes eseten: cim, felek, idopontok, zaradekok.
        Szallitolevel / megrendeles eseten: tetelek (cikkszam, mennyiseg).

        Args:
            filename: A dokumentum fajlneve (pl. "szamla_januar.pdf").

        Visszater: JSON objektum a mezok-ertekek parokkal, vagy hibauzenet
        ha a fajl nem talalhato.
        """
        if context.trace_hook:
            context.trace_hook("get_extraction", {"filename": filename})

        if filename not in context.documents:
            available = list(context.documents.keys())
            return json.dumps(
                {"error": f"Nem talalt dokumentum: '{filename}'",
                 "available": available},
                ensure_ascii=False,
            )

        data = _get_or_extract(context, filename)
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    return get_extraction
