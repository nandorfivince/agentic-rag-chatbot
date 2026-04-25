"""compare_documents tool -- ket dokumentum strukturalt adatainak osszevetese.

Alaplogika:
- Mindket dokumentumot extractaljuk (ha meg nincs cache-ben)
- Osszegeket, datumokat, feleket szinkronizaltan hasonlitjuk
- Tetelek eseten cikkszam-alapu matching + mennyiseg-osszevetes
- Kimenet: JSON {"matches": [...], "differences": [...]} severity jelzessel
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool

from tools import ToolContext
from tools.get_extraction import _get_or_extract
from utils.numbers import coerce_number


def _compare_values(
    field_name: str,
    value_a: Any,
    value_b: Any,
    tolerance_abs: float = 1.0,
    tolerance_pct: float = 0.01,
) -> Optional[dict]:
    """Ket ertek osszevetese; None ha nincs elteres vagy nem osszehasonlithato."""
    if value_a is None and value_b is None:
        return None
    if value_a is None or value_b is None:
        return {
            "field": field_name,
            "severity": "warning",
            "message": f"{field_name}: egyik oldalon hianyzik (A={value_a}, B={value_b})",
        }

    num_a = coerce_number(value_a)
    num_b = coerce_number(value_b)
    if num_a is not None and num_b is not None:
        diff = abs(num_a - num_b)
        if diff <= tolerance_abs or diff / max(abs(num_a), abs(num_b), 1) <= tolerance_pct:
            return None
        return {
            "field": field_name,
            "severity": "critical" if diff / max(abs(num_a), abs(num_b)) > 0.05 else "warning",
            "message": (
                f"{field_name}: {num_a:.2f} vs {num_b:.2f} "
                f"(elteres: {num_b - num_a:+.2f})"
            ),
            "value_a": num_a,
            "value_b": num_b,
        }

    # String osszehasonlitas
    if str(value_a).strip() != str(value_b).strip():
        return {
            "field": field_name,
            "severity": "info",
            "message": f"{field_name}: A='{value_a}' vs B='{value_b}'",
        }
    return None


def _compare_items(items_a: list[dict], items_b: list[dict]) -> list[dict]:
    """Tetelek osszevetese: cikkszam vagy megnevezes alapu matching."""
    differences: list[dict] = []

    def _key(it: dict) -> str:
        return (it.get("cikkszam") or it.get("megnevezes") or "").strip().lower()

    map_a = {_key(it): it for it in items_a}
    map_b = {_key(it): it for it in items_b}

    all_keys = set(map_a) | set(map_b)
    for k in sorted(all_keys):
        if not k:
            continue
        a = map_a.get(k)
        b = map_b.get(k)
        if a and not b:
            differences.append({
                "field": f"tetel[{k}]",
                "severity": "warning",
                "message": f"Tetel csak A-ban: {k}",
            })
            continue
        if b and not a:
            differences.append({
                "field": f"tetel[{k}]",
                "severity": "warning",
                "message": f"Tetel csak B-ben: {k}",
            })
            continue

        # Mennyiseg osszevetes
        qty_a = coerce_number(a.get("mennyiseg"))
        qty_b = coerce_number(b.get("mennyiseg"))
        if qty_a is not None and qty_b is not None and qty_a != qty_b:
            diff = qty_b - qty_a
            severity = "critical" if abs(diff) >= 2 else "warning"
            differences.append({
                "field": f"tetel[{k}].mennyiseg",
                "severity": severity,
                "message": (
                    f"Tetel '{k}' mennyiseg-elteres: A={qty_a}, B={qty_b} "
                    f"(kulonbseg: {diff:+g})"
                ),
                "value_a": qty_a,
                "value_b": qty_b,
            })
    return differences


def build_compare_documents_tool(context: ToolContext) -> BaseTool:
    @tool
    def compare_documents(filename_a: str, filename_b: str) -> str:
        """Ket dokumentum strukturalt adatainak osszevetese: osszegek, tetelek, datumok.

        Hasznald ha a felhasznalo kerdezi:
        - "Hasonlitsd ossze a ket szamlat"
        - "Van-e elteres a szallitolevel es a szamla kozott"
        - "Mennyivel dragabb az egyik?"

        Args:
            filename_a: Az elso dokumentum fajlneve.
            filename_b: A masodik dokumentum fajlneve.

        Visszater: JSON {"summary": str, "differences": [...]}.
        Minden eltereshez tartozik egy severity ("info", "warning", "critical").
        """
        if context.trace_hook:
            context.trace_hook(
                "compare_documents",
                {"filename_a": filename_a, "filename_b": filename_b},
            )

        missing = [f for f in (filename_a, filename_b) if f not in context.documents]
        if missing:
            available = list(context.documents.keys())
            return json.dumps(
                {"error": f"Nem talalt dokumentum(ok): {missing}",
                 "available": available},
                ensure_ascii=False,
            )

        data_a = _get_or_extract(context, filename_a)
        data_b = _get_or_extract(context, filename_b)

        fields_to_compare = [
            "kiallitas_datuma", "teljesites_datuma", "fizetesi_hatarido",
            "netto_vegosszeg", "afa_vegosszeg", "brutto_vegosszeg",
            "kiallito_adoszam", "vevo_adoszam",
            "szallito", "vevo", "datum",
        ]
        differences: list[dict] = []
        for f in fields_to_compare:
            diff = _compare_values(f, data_a.get(f), data_b.get(f))
            if diff:
                differences.append(diff)

        # Tetel-szintu osszevetes
        if isinstance(data_a.get("tetelek"), list) and isinstance(
            data_b.get("tetelek"), list
        ):
            differences.extend(_compare_items(data_a["tetelek"], data_b["tetelek"]))

        # Osszegzo statisztika
        counts = {
            "critical": sum(1 for d in differences if d["severity"] == "critical"),
            "warning": sum(1 for d in differences if d["severity"] == "warning"),
            "info": sum(1 for d in differences if d["severity"] == "info"),
        }
        summary = (
            f"{len(differences)} elteres: {counts['critical']} kritikus, "
            f"{counts['warning']} figyelmeztetes, {counts['info']} informacio."
        )

        return json.dumps(
            {"summary": summary, "counts": counts, "differences": differences},
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return compare_documents
