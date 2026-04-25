"""Szam-parser: HU/EU/US/FR formatumok, penznem-szuffixek, null aliasok.

Celja: az LLM struct. outputban erkezhet "1 234,56 Ft", "1.234,56 EUR",
"1,234.56 USD" -- ezek mindig ugyanaz a szam. A coerce_number() egysegesen
float-ra konvertal, vagy None-t ad ha nem ertelmezheto.

Soha nem dob kivetelt -- hibas input == None (mert a validaciot a rendszer
vegzi, nem itt).
"""

from __future__ import annotations

from typing import Any, Optional

NULL_ALIASES = {"null", "none", "n/a", "na", "-", "--", "–", "—", ""}

# 8 tamogatott penznem kulcsszavai (kis- es nagybetu-insensitiv)
CURRENCY_CODES = {"huf", "ft", "eur", "usd", "gbp", "chf", "czk", "pln", "ron"}
CURRENCY_SYMBOLS = "€$£¥₴₽"


def _strip_currency(s: str) -> str:
    """Penznem-kulcsszo/szimbolum eltavolitasa a vegrol/elejerol."""
    # Szimbolumok
    for sym in CURRENCY_SYMBOLS:
        s = s.replace(sym, "")
    # Kulcsszavak (ket iranyba -- "100 EUR" es "EUR 100")
    lower = s.lower()
    for cur in CURRENCY_CODES:
        if lower.endswith(cur):
            s = s[: len(s) - len(cur)].rstrip(" .,")
            lower = s.lower()
        if lower.startswith(cur):
            s = s[len(cur):].lstrip(" .,")
            lower = s.lower()
    return s.strip()


def coerce_number(value: Any) -> Optional[float]:
    """Barmilyen erteket float-ra alakit, vagy None-nal ter vissza.

    Peldak:
        coerce_number("1 234,56 Ft") -> 1234.56
        coerce_number("1,234.56 USD") -> 1234.56
        coerce_number("-") -> None
        coerce_number(None) -> None
        coerce_number(1234) -> 1234.0
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # Python: bool int-nek minosul -> False-> 0.0, True -> 1.0. Kerdeses hogy
        # ezt akarjuk-e, ezt szandekosan rejecteljuk.
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None
    if s.lower() in NULL_ALIASES:
        return None

    s = _strip_currency(s)
    if not s or s.lower() in NULL_ALIASES:
        return None

    # Whitespace (HU ezres-elvalaszto space + non-breaking space)
    s = s.replace(" ", "").replace(" ", "").replace(" ", "")
    # Zarojelet negativkent ertelmezunk (pl. "(1.234,56)" -> -1234.56)
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]

    if not s:
        return None

    has_dot = "." in s
    has_comma = "," in s

    if has_dot and has_comma:
        # Az utolso forduló jel a tizedes
        if s.rfind(",") > s.rfind("."):
            # HU/EU formatum: "1.234,56" -> 1234.56
            s = s.replace(".", "").replace(",", ".")
        else:
            # US/UK formatum: "1,234.56" -> 1234.56
            s = s.replace(",", "")
    elif has_comma:
        # Csak vesszo -> HU/EU. Ha max 2-3 jegy a vessző utan -> tizedes.
        # Ha 3 jegy es a bal oldal is 3 csoportosan: ezres-elvalaszto.
        parts = s.split(",")
        if (
            len(parts) == 2
            and parts[1].isdigit()
            and len(parts[1]) in (1, 2)
        ):
            s = s.replace(",", ".")
        else:
            # Ezres-elvalaszto feltetelezes (HU: "1,234,567")
            s = s.replace(",", "")
    # has_dot-only: hagyjuk -- valoszinuleg US tizedes vagy EU-ezres.
    # Az utobbira ra lehetne tapogatni, de a kettertelműség miatt alapesetnek
    # a US tizedest hagyjuk; ha probakent 0.123-as hibat latunk, a schema
    # maga fogja kifejezni a kontextust.

    try:
        val = float(s)
    except ValueError:
        return None
    return -val if negative else val
