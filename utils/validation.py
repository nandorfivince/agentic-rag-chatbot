"""Determinisztikus validacio: matek, datum, adoszam CDV, plauzibilitas.

Ezek a kod-szintu ellenorzesek nem LLM-fuggok -- a `validate_document` tool
ezekre epul. A cel: amit Python-bol meg lehet allapitani (nettó+ÁFA=bruttó),
azt ne bizzuk az LLM-re.

A visszateres egyseges: list[dict] a hibak leirasaval.
    [{"type": str, "severity": str, "message": str, ...}, ...]

severity: "alacsony" | "kozepes" | "magas"
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from utils.numbers import coerce_number


# ---------------------------------------------------------------------------
# Adoszam CDV (mod-11 ellenorzo jegy, magyar)
# ---------------------------------------------------------------------------

# Sulyok a magyar adoszam elso 8 szamjegyehez (Art. 22. § szerint)
_HU_TAX_WEIGHTS = [9, 7, 3, 1, 9, 7, 3, 1]


def validate_tax_number(tax_number: Any) -> list[dict]:
    """Magyar adoszam CDV ellenorzes: XXXXXXXX-X-XX format + mod-11 cheksum.

    Az elso 8 szamjegy sulyozott osszege a 9. szamjeg (CDV). Sulyok:
    [9, 7, 3, 1, 9, 7, 3, 1].  checksum = sum(d_i * w_i) mod 10.
    """
    if tax_number is None:
        return []
    s = str(tax_number).strip()
    if not s:
        return []

    # Forma: 8-1-2 (kötőjel opcionálisan elhagyva)
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) != 11:
        return [{
            "type": "adoszam_formatum",
            "severity": "kozepes",
            "message": f"Ervenytelen adoszam hossz: {s!r} (varhato: 11 szamjegy).",
        }]

    first8 = digits[:8]
    cdv_digit = int(digits[8])
    weighted_sum = sum(int(d) * w for d, w in zip(first8, _HU_TAX_WEIGHTS))
    expected_cdv = weighted_sum % 10

    if cdv_digit != expected_cdv:
        return [{
            "type": "adoszam_cdv",
            "severity": "magas",
            "message": (
                f"Adoszam CDV hiba: {s!r}. Varhato 9. szamjegy: {expected_cdv}, "
                f"talalt: {cdv_digit}."
            ),
        }]
    return []


# ---------------------------------------------------------------------------
# Szamla matematikai validacio
# ---------------------------------------------------------------------------

def _nearly_equal(a: float, b: float, tolerance_pct: float = 0.01,
                  abs_tolerance: float = 1.0) -> bool:
    """Kis relatív toleranciával (1%) + abszolút (1 egység) toleranciával osszehasonlit.

    A kerekitesi kulonbsegek elfogadasa (pl. 12345.67 Ft kontra 12345 Ft).
    """
    if a == b:
        return True
    diff = abs(a - b)
    return diff <= abs_tolerance or diff / max(abs(a), abs(b)) <= tolerance_pct


def validate_invoice_math(extracted: dict) -> list[dict]:
    """Szamla matek: netto+afa=brutto, tetelek=vegosszeg, per-tetel matek.

    A schema-t rugalmasan kezeli: az extract-ben a mezok nevei valtozhatnak.
    """
    if not isinstance(extracted, dict):
        return []

    errors: list[dict] = []

    # Foosszeg ellenorzes: netto + afa = brutto
    netto = coerce_number(extracted.get("netto_vegosszeg") or extracted.get("netto"))
    afa = coerce_number(extracted.get("afa_vegosszeg") or extracted.get("afa"))
    brutto = coerce_number(extracted.get("brutto_vegosszeg") or extracted.get("brutto"))

    if netto is not None and afa is not None and brutto is not None:
        expected = netto + afa
        if not _nearly_equal(expected, brutto, abs_tolerance=2.0):
            errors.append({
                "type": "foosszeg_matek",
                "severity": "magas",
                "message": (
                    f"Netto + AFA != Brutto. Netto: {netto:.2f}, AFA: {afa:.2f}, "
                    f"Varhato brutto: {expected:.2f}, talalt: {brutto:.2f} "
                    f"(elteres: {brutto - expected:+.2f})."
                ),
                "expected": expected,
                "actual": brutto,
            })

    # Tetelek osszege vs netto_vegosszeg
    tetelek = extracted.get("tetelek")
    if isinstance(tetelek, list) and tetelek and netto is not None:
        tetel_netto_sum = 0.0
        for t in tetelek:
            if not isinstance(t, dict):
                continue
            t_netto = coerce_number(t.get("netto_osszeg") or t.get("netto"))
            if t_netto is not None:
                tetel_netto_sum += t_netto

        if tetel_netto_sum > 0 and not _nearly_equal(
            tetel_netto_sum, netto, abs_tolerance=5.0
        ):
            errors.append({
                "type": "tetelek_osszege",
                "severity": "magas",
                "message": (
                    f"Tetelek netto osszege {tetel_netto_sum:.2f}, de a "
                    f"netto_vegosszeg {netto:.2f} (elteres: "
                    f"{netto - tetel_netto_sum:+.2f})."
                ),
                "expected": tetel_netto_sum,
                "actual": netto,
            })

    # Per-tetel matek: mennyiseg * egysegar = netto_osszeg
    if isinstance(tetelek, list):
        for idx, t in enumerate(tetelek, 1):
            if not isinstance(t, dict):
                continue
            mennyiseg = coerce_number(t.get("mennyiseg"))
            egysegar = coerce_number(t.get("egysegar_netto") or t.get("egysegar"))
            t_netto = coerce_number(t.get("netto_osszeg") or t.get("netto"))
            if (
                mennyiseg is not None
                and egysegar is not None
                and t_netto is not None
            ):
                expected = mennyiseg * egysegar
                if not _nearly_equal(expected, t_netto, abs_tolerance=1.0):
                    errors.append({
                        "type": "tetel_matek",
                        "severity": "kozepes",
                        "message": (
                            f"{idx}. tetel: {mennyiseg} x {egysegar:.2f} != "
                            f"{t_netto:.2f} (varhato: {expected:.2f})."
                        ),
                        "tetel_index": idx,
                        "expected": expected,
                        "actual": t_netto,
                    })

    # Adoszam CDV (ha van)
    for key in ("kiallito_adoszam", "vevo_adoszam"):
        tax_num = extracted.get(key)
        if tax_num:
            errors.extend(validate_tax_number(tax_num))
        # Nested structure: extracted["kiallito"]["adoszam"]
    for subkey in ("kiallito", "vevo"):
        sub = extracted.get(subkey)
        if isinstance(sub, dict):
            tax_num = sub.get("adoszam")
            if tax_num:
                errors.extend(validate_tax_number(tax_num))

    return errors


# ---------------------------------------------------------------------------
# Datum logika
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y.%m.%d.",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d.%m.%Y.",
    "%d/%m/%Y",
]


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    s = str(value).strip().rstrip(".")
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def validate_date_logic(extracted: dict) -> list[dict]:
    """Datum-sorrend + jovobeli-detekt.

    - teljesites <= kiallitas: ez a normalis sorrend (szolgaltatas elvegzese
      utan kiallitas). Ha teljesites > kiallitas + 90 nap, az gyanus
      eloleszamla / hibasan datalt szamla.
    - fizetesi_hatarido >= kiallitas: korabbi hatarido nyilvanvalo elirasi hiba.
    - jovoben > ma + 2 ev -> gyanus.
    """
    if not isinstance(extracted, dict):
        return []
    errors: list[dict] = []

    kiallitas = _parse_date(extracted.get("kiallitas_datuma"))
    teljesites = _parse_date(extracted.get("teljesites_datuma"))
    fizetesi = _parse_date(extracted.get("fizetesi_hatarido"))

    # Eloleszamla: teljesites tul messze a kiallitas utan (> 90 nap)
    if kiallitas and teljesites:
        gap_days = (teljesites - kiallitas).days
        if gap_days > 90:
            errors.append({
                "type": "datum_sorrend",
                "severity": "kozepes",
                "message": (
                    f"Teljesites datuma ({teljesites}) tul messze a kiallitas "
                    f"({kiallitas}) utan: {gap_days} nap. Eloleszamlanal "
                    f"szokatlan, ellenorzesre szorul."
                ),
            })

    if kiallitas and fizetesi and fizetesi < kiallitas:
        errors.append({
            "type": "datum_sorrend",
            "severity": "kozepes",
            "message": (
                f"Fizetesi hatarido ({fizetesi}) korabbi mint kiallitas "
                f"({kiallitas}) -- elirasi hiba lehet."
            ),
        })

    today = date.today()
    far_future = today.replace(year=today.year + 2)
    for label, d in (("kiallitas", kiallitas), ("teljesites", teljesites),
                    ("fizetesi_hatarido", fizetesi)):
        if d and d > far_future:
            errors.append({
                "type": "datum_jovobeli",
                "severity": "alacsony",
                "message": (
                    f"{label} datuma tul messze a jovoben: {d} "
                    f"(ma: {today}). Elirasi hiba lehet."
                ),
            })

    return errors


# ---------------------------------------------------------------------------
# Plauzibilitas -- szokatlan ertekek
# ---------------------------------------------------------------------------

_STANDARD_VAT_RATES = {0.0, 5.0, 18.0, 27.0}


def validate_plausibility(extracted: dict) -> list[dict]:
    """Szokatlan erteketket jelez -- ezek nem feltetlenul hibak, csak gyanusak."""
    if not isinstance(extracted, dict):
        return []
    warnings: list[dict] = []

    # Negativ vegosszeg
    for key in ("netto_vegosszeg", "afa_vegosszeg", "brutto_vegosszeg"):
        v = coerce_number(extracted.get(key))
        if v is not None and v < 0:
            warnings.append({
                "type": "plauzibilitas_negativ",
                "severity": "kozepes",
                "message": f"Negativ {key}: {v:.2f} -- ellenorzesre szorul.",
            })

    # Vegosszeg > 100 millio
    brutto = coerce_number(extracted.get("brutto_vegosszeg"))
    if brutto is not None and brutto > 100_000_000:
        warnings.append({
            "type": "plauzibilitas_nagy_osszeg",
            "severity": "alacsony",
            "message": f"Szokatlanul nagy brutto vegosszeg: {brutto:,.2f}.",
        })

    # Nem szabvanyos AFA kulcs a tetelekben
    tetelek = extracted.get("tetelek")
    if isinstance(tetelek, list):
        for idx, t in enumerate(tetelek, 1):
            if not isinstance(t, dict):
                continue
            afa_kulcs = coerce_number(t.get("afa_kulcs"))
            if afa_kulcs is not None and afa_kulcs not in _STANDARD_VAT_RATES:
                warnings.append({
                    "type": "plauzibilitas_afa_kulcs",
                    "severity": "alacsony",
                    "message": (
                        f"{idx}. tetel nem szabvanyos AFA kulcs: {afa_kulcs}% "
                        f"(szabvany: 0, 5, 18, 27%)."
                    ),
                })

    return warnings


# ---------------------------------------------------------------------------
# Osszegzo belepes -- a validate_document tool hasznalja
# ---------------------------------------------------------------------------

def validate_all(extracted: dict) -> list[dict]:
    """Minden validacio futtatasa egy dokumentumra."""
    errors: list[dict] = []
    errors.extend(validate_invoice_math(extracted))
    errors.extend(validate_date_logic(extracted))
    errors.extend(validate_plausibility(extracted))
    return errors
