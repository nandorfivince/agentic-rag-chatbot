"""Determinisztikus regex-alapu extrakcio a generalt sablon-dokumentumokhoz.

Mi ez? Az LLM-alapu strukturalt extrakcio idoigenyes es drag az Ollama-n, ra-
adasul dummy LLM-mel nem is muködne. Ezert a szoveges dokumentumbol egy egy-
szeru regex reteggel kinyerjuk a legfontosabb mezoket (szamla_szam, osszegek,
datumok, felek, tetelek) -- igy a tool-ok out-of-the-box mukodnek barhol.

A regex-reteget a `data/generate_samples.py` altal generalt PDF-ek strukturara
hangoltuk, de a minta eleg altalanos hogy hasonlo magyar uzleti doksikon is
jol dolgozzon. Ha valami mezo nem talalhato, None-t adunk vissza -- a rend-
szer robusztusan kezeli a hianyt.

Tipus-tervezes (osszesitve):
    {
        "doc_type": "szamla" | "szerzodes" | "szallitolevel" | "megrendeles" | "egyeb",
        # szamla-specifikus
        "szamla_szam", "kiallitas_datuma", "teljesites_datuma", "fizetesi_hatarido",
        "kiallito", "kiallito_adoszam", "vevo", "vevo_adoszam",
        "netto_vegosszeg", "afa_vegosszeg", "brutto_vegosszeg",
        "tetelek": [{"megnevezes", "mennyiseg", "egysegar_netto",
                     "netto_osszeg", "afa_kulcs"}, ...]
        # szerzodes-specifikus
        "cim", "felek", "kezdet", "lejarat", "havi_dij", "zaradekok": [str, ...]
        # szallitolevel/megrendeles
        "dokumentum_szam", "datum", "szallito", "szallito_adoszam",
        "vevo", "vevo_adoszam",
        "tetelek": [{"cikkszam", "megnevezes", "mennyiseg"}, ...]
    }
"""

from __future__ import annotations

import re
from typing import Optional

from utils.numbers import coerce_number


# ---------------------------------------------------------------------------
# Tipus-detekcio
# ---------------------------------------------------------------------------

def classify_heuristic(text: str) -> str:
    """Dokumentumtipus heurisztikus meghatarozasa a szoveg elso 500 karakterebol."""
    head = text[:500].lower()
    # Prioritas: az elsokent talalt tipus nyer (cim a dokumentum elejen)
    if re.search(r"\bszámla\b|\bszamla\b|\binvoice\b", head):
        return "szamla"
    if re.search(r"\bszállítólevél\b|\bszallitolevel\b|\bdelivery note\b", head):
        return "szallitolevel"
    if re.search(r"\bmegrendelés\b|\bmegrendeles\b|\bpurchase order\b", head):
        return "megrendeles"
    if re.search(r"\bszerződés\b|\bszerzodes\b|\bnda\b|\bmegállapodás\b|"
                 r"\bcontract\b|\bagreement\b", head):
        return "szerzodes"
    return "egyeb"


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _field(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags)
    if not m:
        return None
    return m.group(1).strip().rstrip(".,;:")


def _amount(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return coerce_number(m.group(1))


# ---------------------------------------------------------------------------
# Szamla
# ---------------------------------------------------------------------------

# Egy tetel-blokk a szamlan -- a PyMuPDF kimeneten minden mezo kulon sor:
#   "<megnevezes>\n<mennyiseg>\n<egysegar> Ft\n<netto> Ft\n<afa>%"
# A \s a whitespace-osztaly includalja a newline-t es a NBSP-t is (U+00A0, U+202F).
_INVOICE_ITEM_RE = re.compile(
    r"^(?P<megnevezes>[^\n\d][^\n]{2,79})\n"
    r"(?P<mennyiseg>\d+)\n"
    r"(?P<egysegar>[\d,.\s]+?)\s*Ft\n"
    r"(?P<netto>[\d,.\s]+?)\s*Ft\n"
    r"(?P<afa>\d{1,2})\s*%",
    re.MULTILINE,
)


def _extract_invoice_items(text: str) -> list[dict]:
    items: list[dict] = []
    for m in _INVOICE_ITEM_RE.finditer(text):
        items.append({
            "megnevezes": m.group("megnevezes").strip(),
            "mennyiseg": coerce_number(m.group("mennyiseg")),
            "egysegar_netto": coerce_number(m.group("egysegar")),
            "netto_osszeg": coerce_number(m.group("netto")),
            "afa_kulcs": coerce_number(m.group("afa")),
        })
    return items


def _extract_invoice(text: str) -> dict:
    kiallito_block = _field(text, r"Kiállító\s*\n+\s*([^\n]+)")
    vevo_block = _field(text, r"Vevő\s*\n+\s*([^\n]+)")

    # Adoszamok a "Kiallito" / "Vevo" blokkokban
    kiallito_adoszam = _field(
        text, r"Kiállító[\s\S]{0,200}?Adószám:\s*([\d\-]+)"
    )
    vevo_adoszam = _field(
        text, r"Vevő[\s\S]{0,200}?Adószám:\s*([\d\-]+)"
    )

    return {
        "doc_type": "szamla",
        "szamla_szam": _field(text, r"Számla száma:\s*([^\n]+)"),
        "kiallitas_datuma": _field(text, r"Kiállítás dátuma:\s*([\d.\-/]+)"),
        "teljesites_datuma": _field(text, r"Teljesítés dátuma:\s*([\d.\-/]+)"),
        "fizetesi_hatarido": _field(text, r"Fizetési határidő:\s*([\d.\-/]+)"),
        "kiallito": kiallito_block,
        "kiallito_adoszam": kiallito_adoszam,
        "vevo": vevo_block,
        "vevo_adoszam": vevo_adoszam,
        "netto_vegosszeg": _amount(
            text, r"Nettó végösszeg\s*\n?\s*([\d,.\s]+?)\s*Ft"
        ),
        # "AFA osszesen (27%)" sor, utana newline, utana az osszeg + Ft.
        # A "(27%)" zarojelet atugorjuk [^\n]*\n\s* mintaval.
        "afa_vegosszeg": _amount(
            text, r"ÁFA\s+összesen[^\n]*\n\s*([\d,.\s]+?)\s*Ft"
        ),
        "brutto_vegosszeg": _amount(
            text, r"Bruttó végösszeg\s*\n?\s*([\d,.\s]+?)\s*Ft"
        ),
        "tetelek": _extract_invoice_items(text),
    }


# ---------------------------------------------------------------------------
# Szerzodes
# ---------------------------------------------------------------------------

def _extract_contract(text: str) -> dict:
    # Cim: az elso sor (H1)
    first_lines = text.strip().split("\n", 2)
    cim = first_lines[0].strip() if first_lines else None

    # Zaradek-sorok: "1. Cim", "2. Cim" stb. (soronkent, csak a sor elejen)
    zaradekok = re.findall(r"^\d+\.\s+([^\n]+)$", text, re.MULTILINE)

    havi_dij = _field(text, r"[Hh]avi [dD]íj:?\s*([^\n]+)")

    return {
        "doc_type": "szerzodes",
        "cim": cim,
        "felek": _field(text, r"Felek:\s*([\s\S]+?)\n\s*Hatály"),
        "kezdet": _field(text, r"Hatály kezdete:\s*([\d.\-/]+)"),
        "lejarat": _field(text, r"Lejárat:\s*([\d.\-/]+)"),
        "havi_dij": havi_dij,
        "zaradekok": zaradekok,
    }


# ---------------------------------------------------------------------------
# Szallitolevel / Megrendeles
# ---------------------------------------------------------------------------

# Egy tetel-blokk: "<cikkszam>\n<megnevezes>\n<mennyiseg> <mertek>"
# A generalt PDF-eken minden mezo kulon sor, majd a mennyiseg+mertek egyutt.
_DELIVERY_ITEM_RE = re.compile(
    r"^(?P<cikkszam>[A-Z]{2,4}-\d{2,5})\s*\n"
    r"(?P<megnevezes>[^\n]+)\s*\n"
    r"(?P<mennyiseg>\d+)\s*(?P<mertek>db|kg|m|m2|m3|l)\b",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_delivery_items(text: str) -> list[dict]:
    items: list[dict] = []
    for m in _DELIVERY_ITEM_RE.finditer(text):
        items.append({
            "cikkszam": m.group("cikkszam").strip(),
            "megnevezes": m.group("megnevezes").strip(),
            "mennyiseg": coerce_number(m.group("mennyiseg")),
            "mertek": m.group("mertek").lower(),
        })
    return items


def _extract_delivery_or_order(text: str, doc_type: str) -> dict:
    szallito_block = _field(text, r"Szállító\s*\n+\s*([^\n]+)")
    vevo_block = _field(text, r"Vevő\s*\n+\s*([^\n]+)")
    szallito_adoszam = _field(
        text, r"Szállító[\s\S]{0,200}?Adószám:\s*([\d\-]+)"
    )
    vevo_adoszam = _field(
        text, r"Vevő[\s\S]{0,200}?Adószám:\s*([\d\-]+)"
    )
    megjegyzes = _field(text, r"Megjegyzés:\s*([\s\S]+)")

    return {
        "doc_type": doc_type,
        "dokumentum_szam": _field(text, r"Dokumentum száma:\s*([^\s\n]+)"),
        "datum": _field(text, r"Dátum:\s*([\d.\-/]+)"),
        "szallito": szallito_block,
        "szallito_adoszam": szallito_adoszam,
        "vevo": vevo_block,
        "vevo_adoszam": vevo_adoszam,
        "tetelek": _extract_delivery_items(text),
        "megjegyzes": megjegyzes,
    }


# ---------------------------------------------------------------------------
# Osszesito belepes
# ---------------------------------------------------------------------------

def extract_structured(text: str) -> dict:
    """Doksi szovegbol kivonja a strukturalt mezoket (regex-alapu)."""
    doc_type = classify_heuristic(text)
    if doc_type == "szamla":
        return _extract_invoice(text)
    if doc_type == "szerzodes":
        return _extract_contract(text)
    if doc_type in ("szallitolevel", "megrendeles"):
        return _extract_delivery_or_order(text, doc_type)
    return {"doc_type": "egyeb"}
