"""Intent classifier node -- keyword-alapu, LLM-fuggetlen.

Miert keyword-alapu? Az intent-detektalas a workflow elso lepese, gyors es
egyertelmu kell legyen. Egy LLM-hivas itt 1-5 sec-et adhat a feldolgozashoz,
ezzel szemben a regex under 1ms. Mindket LLM provider-rel (Ollama, Dummy)
egyarant mukodik -- ez teszi az intent detektalast robusztussa.

A 6 intent kategoria:
    - "list"     : "hany dokumentum van?", "milyen fajlok?"
    - "extract"  : "mennyi a brutto?", "mikor az esedekes?"
    - "search"   : "milyen klauzulak?", "keress rákot"
    - "compare"  : "hasonlitsd ossze", "van elteres?"
    - "validate" : "van-e hiba?", "ellenorizd a matekot"
    - "chat"     : default -- altalanos beszelgetes (nincs tool-szukseglet)
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from graph.state import AgentState
from utils.trace import trace_append


# A regex-mintak szandekosan egyszeruek es breite, hogy a magyar kerdezesek
# termeszetes valtozatait is lefedjek. A sorrend prioritasos: az elso talalat
# nyer.
_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(hiba|hibat|hibas|validal|ellenoriz|matematik|szamit)", re.I),
     "validate"),
    (re.compile(r"\b(hasonlit|osszevet|elter|kulonbs|mennyivel|dragabb|olcso)", re.I),
     "compare"),
    (re.compile(r"\b(osszeg|brutto|netto|afa|datum|kiallit|vevo|fizetesi|tetel|hatarido)", re.I),
     "extract"),
    (re.compile(r"\b(keres|tartalm|szerepel|klauzul|milyen \w+ klauzul|melyik|hol)", re.I),
     "search"),
    (re.compile(r"\b(hany dokumentum|milyen dokumentum|feltoltve|listaz|dokumentumok|fajlok)", re.I),
     "list"),
]


def _last_user_text(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
    return ""


def classify_intent(text: str) -> str:
    """Ures input -> "chat"; kulonben az elso illeszkedo minta nyer."""
    if not text.strip():
        return "chat"
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return "chat"


def intent_classifier_node(state: AgentState) -> dict:
    """Node belepes: kinyeri az utolso user-uzenet intent-jet."""
    text = _last_user_text(state.get("messages", []))
    intent = classify_intent(text)
    trace = trace_append(state, "1. intent_classifier", f"intent={intent}")
    return {"intent": intent, "trace": trace}
