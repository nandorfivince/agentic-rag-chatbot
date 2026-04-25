"""Szintetikus minta PDF-ek generalasa a tesztekhez + demonstaciohoz.

8 dokumentum keszul:

Szamla sorozat (audit-demo, gyanus arnovekedes):
  - szamla_januar.pdf     (BudaData Kft. -> CsimpiTech Bt., alap ar)
  - szamla_februar.pdf    (+5% az elozohoz kepest)
  - szamla_marcius.pdf    (+50% a januari arhoz kepest, riasztasi jelzes)

Szerzodesek (DD-demo):
  - nda_smartsensors.pdf           (titoktartasi megallapodas)
  - szolgaltatasi_szerzodes_datalab.pdf  (havi 2M Ft, 12 honap)

Kereszthatas haromlevel (compare-demo):
  - megrendeles_epitokezi.pdf      (40 db HI-100 I-gerenda)
  - szallitolevel_epitokezi.pdf    (38 db szallitva -- 2 db hiany)
  - szamla_epitokezi.pdf           (40 db szamlazva -- 2 db tulszamlazas)

Futtatas:  python data/generate_samples.py
Kimenet:   data/sample_docs/*.pdf
"""

from __future__ import annotations

from pathlib import Path

import fitz

OUTPUT_DIR = Path(__file__).parent / "sample_docs"


# ---------------------------------------------------------------------------
# Mod-11 CDV szamitas (Art. 22. § -- megegyezik utils/validation.py-vel)
# ---------------------------------------------------------------------------

_HU_TAX_WEIGHTS = [9, 7, 3, 1, 9, 7, 3, 1]


def _compute_cdv(first8: str) -> int:
    """Magyar adoszam mod-11 CDV szamitas az elso 8 szamjegybol."""
    return sum(int(d) * w for d, w in zip(first8, _HU_TAX_WEIGHTS)) % 10


def _tax(first8: str, megye_kod: str = "42") -> str:
    """Egyszeru segito: 'XXXXXXXX-C-MM', ahol C az automatikusan szamolt CDV."""
    cdv = _compute_cdv(first8)
    return f"{first8}-{cdv}-{megye_kod}"


# Generalt erveny adoszamok (mod-11 CDV-vel, lekerdezett megyekoddal)
TAX_CSIMPITECH = _tax("12345678", "42")     # 12345678-2-42 (eredetileg is helyes)
TAX_BUDADATA = _tax("98765432", "41")       # 98765432-8-41 (kijavitva)
TAX_DATALAB = _tax("24680246", "42")        # 24680246-4-42 (kijavitva)
TAX_EPITOKEZI = _tax("11223344", "13")      # 11223344-8-13 (kijavitva)
TAX_VAREPITO = _tax("55667788", "42")       # 55667788-8-42 (kijavitva)


def _render_html_pdf(output_path: Path, html: str) -> None:
    """HTML -> PDF egyetlen A4 oldalra (PyMuPDF insert_htmlbox, UTF-8)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    rect = fitz.Rect(40, 40, 555, 802)
    # Stilus CSS-t hozzaadunk hogy legyenek olvashato margok
    full_html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><style>
        body {{ font-family: sans-serif; font-size: 10pt; color: #000; }}
        h1 {{ font-size: 18pt; margin: 0 0 8pt 0; }}
        h2 {{ font-size: 12pt; margin: 12pt 0 4pt 0; }}
        p  {{ margin: 4pt 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 6pt 0; }}
        th, td {{ border: 1px solid #444; padding: 3pt 5pt; text-align: left; }}
        th {{ background: #ddd; }}
        .right {{ text-align: right; }}
        .total {{ font-weight: bold; background: #eef; }}
    </style></head><body>{html}</body></html>"""
    page.insert_htmlbox(rect, full_html)
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()


# ---------------------------------------------------------------------------
# Szamla sablon (audit-demo: 3 szamla novekvo arakkal)
# ---------------------------------------------------------------------------

def _invoice_html(
    szamla_szam: str,
    kiallitas: str,
    teljesites: str,
    fizetesi_hatarido: str,
    kiallito_nev: str,
    kiallito_adoszam: str,
    kiallito_cim: str,
    vevo_nev: str,
    vevo_adoszam: str,
    vevo_cim: str,
    tetelek: list[dict],
) -> str:
    netto_osszeg = sum(t["netto_osszeg"] for t in tetelek)
    afa_osszeg = sum(t["netto_osszeg"] * (t["afa_kulcs"] / 100) for t in tetelek)
    brutto_osszeg = netto_osszeg + afa_osszeg

    rows = "\n".join(
        f"<tr><td>{t['megnevezes']}</td>"
        f"<td class=\"right\">{t['mennyiseg']}</td>"
        f"<td class=\"right\">{t['egysegar']:,.0f} Ft</td>"
        f"<td class=\"right\">{t['netto_osszeg']:,.0f} Ft</td>"
        f"<td class=\"right\">{t['afa_kulcs']:.0f}%</td></tr>"
        for t in tetelek
    )

    return f"""
    <h1>SZÁMLA</h1>
    <p><b>Számla száma:</b> {szamla_szam}</p>
    <p><b>Kiállítás dátuma:</b> {kiallitas} &nbsp;&nbsp;
       <b>Teljesítés dátuma:</b> {teljesites} &nbsp;&nbsp;
       <b>Fizetési határidő:</b> {fizetesi_hatarido}</p>

    <h2>Kiállító</h2>
    <p>{kiallito_nev}<br/>
       Adószám: {kiallito_adoszam}<br/>
       Cím: {kiallito_cim}</p>

    <h2>Vevő</h2>
    <p>{vevo_nev}<br/>
       Adószám: {vevo_adoszam}<br/>
       Cím: {vevo_cim}</p>

    <h2>Tételek</h2>
    <table>
        <tr><th>Megnevezés</th><th class="right">Mennyiség</th>
            <th class="right">Egységár</th><th class="right">Nettó</th>
            <th class="right">ÁFA</th></tr>
        {rows}
    </table>

    <table>
        <tr><td><b>Nettó végösszeg</b></td>
            <td class="right">{netto_osszeg:,.0f} Ft</td></tr>
        <tr><td><b>ÁFA összesen (27%)</b></td>
            <td class="right">{afa_osszeg:,.0f} Ft</td></tr>
        <tr class="total"><td><b>Bruttó végösszeg</b></td>
            <td class="right">{brutto_osszeg:,.0f} Ft</td></tr>
    </table>

    <p style="font-size:8pt;color:#666">A számla elektronikusan kiállítva
       és érvényes aláírás nélkül. Pénznem: HUF.</p>
    """


def _generate_invoice_series() -> None:
    """3 szamla: januar -> februar (+5%) -> marcius (+50% a januaritol).

    Az elso ket szamla normalis dragulasa (5%), a harmadik ugras szemelytelenitheto
    auditalasi riasztaskent -- ezt a dummy LLM compare_documents toolja kell
    kimutassa.
    """
    common = dict(
        kiallito_nev="CsimpiTech Bt.",
        kiallito_adoszam=TAX_CSIMPITECH,
        kiallito_cim="1137 Budapest, Szent István krt. 12.",
        vevo_nev="BudaData Kft.",
        vevo_adoszam=TAX_BUDADATA,
        vevo_cim="1095 Budapest, Lechner Ödön fasor 9.",
    )

    # Januar -- alapar
    base_price = 50_000  # Ft/ora
    january_hours = 40
    _render_html_pdf(OUTPUT_DIR / "szamla_januar.pdf", _invoice_html(
        szamla_szam="2026/001",
        kiallitas="2026.01.31",
        teljesites="2026.01.30",
        fizetesi_hatarido="2026.02.29",
        tetelek=[
            {"megnevezes": "Szoftverfejlesztési szolgáltatás",
             "mennyiseg": january_hours, "egysegar": base_price,
             "netto_osszeg": january_hours * base_price, "afa_kulcs": 27},
        ],
        **common,
    ))

    # Februar -- 5% dragabb (normal)
    feb_price = int(base_price * 1.05)
    feb_hours = 42
    _render_html_pdf(OUTPUT_DIR / "szamla_februar.pdf", _invoice_html(
        szamla_szam="2026/002",
        kiallitas="2026.02.28",
        teljesites="2026.02.27",
        fizetesi_hatarido="2026.03.30",
        tetelek=[
            {"megnevezes": "Szoftverfejlesztési szolgáltatás",
             "mennyiseg": feb_hours, "egysegar": feb_price,
             "netto_osszeg": feb_hours * feb_price, "afa_kulcs": 27},
        ],
        **common,
    ))

    # Marcius -- 50% dragabb (gyanus)
    mar_price = int(base_price * 1.50)
    mar_hours = 44
    _render_html_pdf(OUTPUT_DIR / "szamla_marcius.pdf", _invoice_html(
        szamla_szam="2026/003",
        kiallitas="2026.03.31",
        teljesites="2026.03.29",
        fizetesi_hatarido="2026.04.30",
        tetelek=[
            {"megnevezes": "Szoftverfejlesztési szolgáltatás",
             "mennyiseg": mar_hours, "egysegar": mar_price,
             "netto_osszeg": mar_hours * mar_price, "afa_kulcs": 27},
        ],
        **common,
    ))


# ---------------------------------------------------------------------------
# Szerzodes sablonok
# ---------------------------------------------------------------------------

def _contract_html(
    cim: str,
    felek: str,
    kezdet: str,
    lejarat: str,
    zaradekok: list[tuple[str, str]],  # [(cim, szoveg), ...]
    havi_dij: str | None = None,
) -> str:
    dij_html = (f"<p><b>Havi díj:</b> {havi_dij}</p>" if havi_dij else "")
    zaradek_html = "\n".join(
        f"<h2>{i+1}. {c}</h2><p>{sz}</p>"
        for i, (c, sz) in enumerate(zaradekok)
    )
    return f"""
    <h1>{cim.upper()}</h1>
    <p><b>Felek:</b> {felek}</p>
    <p><b>Hatály kezdete:</b> {kezdet} &nbsp;&nbsp;
       <b>Lejárat:</b> {lejarat}</p>
    {dij_html}
    {zaradek_html}
    <p style="font-size:8pt;color:#666">Budapest, {kezdet}. Mindkét fél képviselője
       aláírásával érvényesíti.</p>
    """


def _generate_contracts() -> None:
    # NDA
    _render_html_pdf(OUTPUT_DIR / "nda_smartsensors.pdf", _contract_html(
        cim="Titoktartási Megállapodás (NDA)",
        felek=(
            "SmartSensors Zrt. (adószám: " + _tax("13579246", "13") + ", "
            "1113 Budapest, Bartók Béla út 152.) és "
            "InfoTech Kft. (adószám: " + _tax("86420135", "42") + ", "
            "1051 Budapest, Sas utca 8.)"
        ),
        kezdet="2026.01.15",
        lejarat="2027.01.15",
        zaradekok=[
            ("Bizalmas információk köre",
             "Minden olyan technikai, üzleti, pénzügyi adat, amelyet a felek "
             "a megállapodás keretében közölnek, beleértve a szoftverspecifikációkat, "
             "ügyféllistákat és árazási modelleket."),
            ("Titoktartási kötelezettség időtartama",
             "A Fogadó Fél a közölt információkat a megállapodás lejáratát követő "
             "5 évig köteles bizalmasan kezelni."),
            ("Kötbér",
             "A kötelezettség megsértése esetén a Fogadó Fél 5.000.000 Ft "
             "kötbér megfizetésére köteles esetenként."),
            ("Kivételek",
             "Nem minősül bizalmas információnak, amely közismert, a Fogadó Fél "
             "részére már korábban ismert volt, vagy hatósági felhívásra kerül kiadásra."),
            ("Irányadó jog és bíróság",
             "A megállapodásra a magyar jog az irányadó. Jogviták esetén a Fővárosi "
             "Törvényszék kizárólagos illetékességét kötik ki a felek."),
        ],
    ))

    # Szolgaltatasi szerzodes
    _render_html_pdf(
        OUTPUT_DIR / "szolgaltatasi_szerzodes_datalab.pdf",
        _contract_html(
            cim="Szoftverszolgáltatási Szerződés",
            felek=(
                f"DataLab Kft. (adószám: {TAX_DATALAB}, 1117 Budapest, "
                f"Budafoki út 60.) mint Szolgáltató, és "
                f"BudaData Kft. (adószám: {TAX_BUDADATA}, 1095 Budapest, "
                f"Lechner Ödön fasor 9.) mint Megbízó"
            ),
            kezdet="2026.02.01",
            lejarat="2027.01.31",
            havi_dij="2.000.000 Ft + 27% ÁFA (bruttó 2.540.000 Ft)",
            zaradekok=[
                ("Szolgáltatás tárgya",
                 "Felhőalapú adatelemző platform üzemeltetése, havi 99.5% SLA "
                 "rendelkezésreállással és 24/7 emelt szintű támogatással."),
                ("Díjazás",
                 "A Megbízó havi 2.000.000 Ft + ÁFA átalánydíjat fizet. "
                 "Egyenleg kiegyenlítése a számla kiállításától számított 30 napon belül."),
                ("Change of control",
                 "Amennyiben a Szolgáltatóban 50%-ot meghaladó tulajdonosi változás "
                 "következik be, a Megbízó jogosult a szerződést azonnali hatállyal "
                 "felmondani, és a már teljesített részteljesítésre eső díjat követelheti vissza."),
                ("Automatikus megújulás",
                 "A szerződés automatikusan további egy év időtartamra megújul, "
                 "kivéve, ha bármely fél a lejárat előtt legalább 60 nappal "
                 "írásban felmondást jelez be."),
                ("Kötbér késedelem esetén",
                 "Az SLA vállalt rendelkezésreállás minden 1%-os hiányosságáért a "
                 "Szolgáltató 100.000 Ft kötbért fizet a Megbízónak."),
                ("Irányadó jog",
                 "A szerződésre a magyar Ptk. rendelkezései vonatkoznak."),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Keresztellenorzes haromlevel: megrendeles / szallitolevel / szamla
# ---------------------------------------------------------------------------

def _order_or_delivery_html(
    cim: str,
    dokumentum_szam: str,
    datum: str,
    szallito: str,
    szallito_adoszam: str,
    vevo: str,
    vevo_adoszam: str,
    tetelek: list[dict],
    megjegyzes: str = "",
) -> str:
    rows = "\n".join(
        f"<tr><td>{t['cikkszam']}</td><td>{t['megnevezes']}</td>"
        f"<td class=\"right\">{t['mennyiseg']} {t['mertek']}</td>"
        f"<td class=\"right\">{t.get('egysegar', '-')}</td></tr>"
        for t in tetelek
    )
    megj = f"<p><b>Megjegyzés:</b> {megjegyzes}</p>" if megjegyzes else ""
    return f"""
    <h1>{cim.upper()}</h1>
    <p><b>Dokumentum száma:</b> {dokumentum_szam} &nbsp;&nbsp;
       <b>Dátum:</b> {datum}</p>

    <h2>Szállító</h2>
    <p>{szallito}<br/>Adószám: {szallito_adoszam}</p>
    <h2>Vevő</h2>
    <p>{vevo}<br/>Adószám: {vevo_adoszam}</p>

    <h2>Tételek</h2>
    <table>
        <tr><th>Cikkszám</th><th>Megnevezés</th>
            <th class="right">Mennyiség</th><th class="right">Egységár</th></tr>
        {rows}
    </table>
    {megj}
    """


def _generate_multi_doc_triplet() -> None:
    """Megrendeles: 40 db HI-100 I-gerenda, 18 500 Ft/db.
    Szallitolevel: 38 db (2 db hiany).
    Szamla: 40 db szamlazva -> 37 000 Ft netto tulszamlazas.
    """
    szallito_name = "ÉpítőKézi Zrt."
    szallito_adoszam = TAX_EPITOKEZI
    vevo_name = "Vár-Építő Kft."
    vevo_adoszam = TAX_VAREPITO

    # 1. Megrendeles
    _render_html_pdf(OUTPUT_DIR / "megrendeles_epitokezi.pdf", _order_or_delivery_html(
        cim="Megrendelés",
        dokumentum_szam="MR-2026/0412",
        datum="2026.04.01",
        szallito=szallito_name,
        szallito_adoszam=szallito_adoszam,
        vevo=vevo_name,
        vevo_adoszam=vevo_adoszam,
        tetelek=[
            {"cikkszam": "HI-100", "megnevezes": "I-gerenda (6 m)",
             "mennyiseg": 40, "mertek": "db", "egysegar": "18 500 Ft/db"},
            {"cikkszam": "HI-050", "megnevezes": "Csavar készlet M16",
             "mennyiseg": 160, "mertek": "db", "egysegar": "420 Ft/db"},
        ],
        megjegyzes="Szállítás határideje: 2026.04.15. Szállítási cím: Vár-Építő raktár, Budapest XXII.",
    ))

    # 2. Szallitolevel -- 38 db (2 db hiany)
    _render_html_pdf(OUTPUT_DIR / "szallitolevel_epitokezi.pdf", _order_or_delivery_html(
        cim="Szállítólevél",
        dokumentum_szam="SZL-2026/0415",
        datum="2026.04.14",
        szallito=szallito_name,
        szallito_adoszam=szallito_adoszam,
        vevo=vevo_name,
        vevo_adoszam=vevo_adoszam,
        tetelek=[
            {"cikkszam": "HI-100", "megnevezes": "I-gerenda (6 m)",
             "mennyiseg": 38, "mertek": "db", "egysegar": "-"},
            {"cikkszam": "HI-050", "megnevezes": "Csavar készlet M16",
             "mennyiseg": 160, "mertek": "db", "egysegar": "-"},
        ],
        megjegyzes=(
            "Megrendelési hivatkozás: MR-2026/0412. "
            "HI-100 I-gerenda: készlethiány miatt 38 db került kiszállításra "
            "a 40 db megrendeltből. A maradék 2 db a következő szállítással érkezik."
        ),
    ))

    # 3. Szamla -- 40 db szamlazva (2 db tulszamlazas)
    netto_hi100 = 40 * 18_500      # 740 000 Ft
    netto_csavar = 160 * 420       # 67 200 Ft
    netto_sum = netto_hi100 + netto_csavar  # 807 200 Ft
    afa_sum = netto_sum * 0.27     # 217 944
    brutto_sum = netto_sum + afa_sum  # 1 025 144
    _render_html_pdf(OUTPUT_DIR / "szamla_epitokezi.pdf", _invoice_html(
        szamla_szam="2026/EP-0418",
        kiallitas="2026.04.18",
        teljesites="2026.04.14",
        fizetesi_hatarido="2026.05.18",
        kiallito_nev=szallito_name,
        kiallito_adoszam=szallito_adoszam,
        kiallito_cim="1221 Budapest, Építő utca 1.",
        vevo_nev=vevo_name,
        vevo_adoszam=vevo_adoszam,
        vevo_cim="1221 Budapest, Nagytétényi út 190.",
        tetelek=[
            {"megnevezes": "HI-100 I-gerenda (6 m)",
             "mennyiseg": 40, "egysegar": 18_500,
             "netto_osszeg": netto_hi100, "afa_kulcs": 27},
            {"megnevezes": "HI-050 Csavar készlet M16",
             "mennyiseg": 160, "egysegar": 420,
             "netto_osszeg": netto_csavar, "afa_kulcs": 27},
        ],
    ))


# ---------------------------------------------------------------------------
# Belepes
# ---------------------------------------------------------------------------

def generate_samples() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _generate_invoice_series()
    _generate_contracts()
    _generate_multi_doc_triplet()

    files = sorted(OUTPUT_DIR.glob("*.pdf"))
    print(f"\n{len(files)} PDF generalva a kovetkezo helyen: {OUTPUT_DIR}")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    generate_samples()
