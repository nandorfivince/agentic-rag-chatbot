"""Dummy LLM provider -- determinisztikus stub valaszokkal.

Miert van ez? A medior feladatkiiras megengedi a dummy LLM-et ("Amennyiben ez
nem lehetseges, dummy LLM-ek hasznalata is elfogadott."). Ez biztositja, hogy
az ertekelo az Ollama telepitese nelkul is ki tudja probalni a rendszert es
lathassa a LangGraph workflow-t, agentic tool-hivast, RAG subgraph-ot mukodes
kozben.

Kulcs tulajdonsagok:
- BaseChatModel subclass, igy LangGraph/LangChain natively kezeli
- bind_tools() tamogatott, a meghivott tool-ok listaja eltarolodik
- Tool-valasztas egyszeru kulcsszo-minta alapjan (rule-based router)
- Max 4 tool-hivas utan determinisztikusan szintezalt valasz
- Deterministikus, reprodukalhato ki-/bemenet (load testhez ideal)

Nem tudomanyos AI, de hitelessen demonstralja a workflow mindegyik agat.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field


# Kulcsszo -> tool nev mapping, prioritasos sorrendben (elso talalat nyer).
# Ezek magyar & angol kulcsszavakra figyelnek, mert a kerdesek magyarul erkeznek.
_INTENT_RULES: list[tuple[re.Pattern, str, dict]] = [
    # validate_document -- hiba, ellenorzes, valid, ervenyes, adoszam
    (re.compile(r"(hiba|hibat|hibas|validal|ellenoriz|matematik|szamit|"
                r"ervenyes|érvényes|adoszam|adószám)", re.I),
     "validate_document", {}),
    # compare_documents -- hasonlit, elter, osszevet, kulonbseg, mennyivel
    (re.compile(r"(hasonlit|hasonlít|osszevet|összevet|elter|eltér|kulonbs|"
                r"különbs|mennyivel|dragabb|drágább|olcso|olcsó)", re.I),
     "compare_documents", {}),
    # get_extraction -- osszeg, afa, brutto, netto, datum, kiallit, ki allit, bocsat
    (re.compile(r"(osszeg|összeg|afa|áfa|brutto|bruttó|netto|nettó|datum|dátum|"
                r"kiallit|kiállít|\bki\s+allit|\bki\s+állít|bocsat|bocsát|"
                r"vevo|vevő|hatarido|határidő|tetel|tétel|fizetesi|fizetési)", re.I),
     "get_extraction", {}),
    # search_documents -- keres, tartalmaz, szerepel, klauzul
    (re.compile(r"(keres|tartalm|szerepel|klauzul|zaradék|záradék|melyik|"
                r"hol van|holvan)", re.I),
     "search_documents", {}),
    # list_documents -- hany, milyen dokumentum, feltoltve, listaz
    (re.compile(r"(hany dokumentum|hány dokumentum|milyen dokumentum|"
                r"milyen fajl|milyen fájl|feltoltve|feltöltve|listaz|listáz|"
                r"dokumentumok|fajlok|fájlok)", re.I),
     "list_documents", {}),
]


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    """Az utolso emberi uzenet szovege -- erre epul a tool-valasztas."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            # Multimodal content list, csak a szoveges reszeket vesszuk
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
    return ""


def _tool_messages(messages: Sequence[BaseMessage]) -> List[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _extract_filename(text: str, docs_hint: Optional[list[str]] = None) -> Optional[str]:
    """Fajlnev kiemelese a kerdesbol.

    Egyszeru heurisztika: keressuk a ".pdf" vagy ".docx" vegzodest, vagy
    ha a docs_hint adott, nezzuk meg hogy melyik feltoltott fajl nevenek
    egy reszlete szerepel a szovegben.
    """
    m = re.search(r"([\w\-]+\.(?:pdf|docx|PDF|DOCX))", text)
    if m:
        return m.group(1)
    if docs_hint:
        lower = text.lower()
        for name in docs_hint:
            stem = name.rsplit(".", 1)[0].lower()
            if stem in lower:
                return name
    return None


def _filenames_from_query(text: str, docs_hint: Optional[list[str]]) -> list[str]:
    """Osszes PDF/DOCX fajlnev a kerdesbol, sorrendben.

    Elso lepes: explicit kiterjesztes-alapu match. Masodik: a docs_hint
    elemei kozul azok, amelyek toveinek reszlete szerepel a szovegben
    (pl. "januari" -> "szamla_januar.pdf", "marciusi" -> "szamla_marcius.pdf").
    """
    results: list[str] = []
    for m in re.finditer(r"([\w\-]+\.(?:pdf|docx|PDF|DOCX))", text):
        n = m.group(1)
        if n not in results:
            results.append(n)
    if docs_hint:
        lower = text.lower()
        for name in docs_hint:
            if name in results:
                continue
            # "szamla_januar.pdf" -> "szamla_januar" -> reszletet keressunk
            stem = name.rsplit(".", 1)[0].lower()
            # Bontunk alkatreszekre: "szamla_januar" -> ["szamla", "januar"]
            parts = [p for p in re.split(r"[_\-\s]+", stem) if len(p) > 3]
            if any(p in lower for p in parts):
                results.append(name)
    return results


def _choose_tool(
    user_text: str,
    bound_tools: list,
    previous_tools_called: list[str],
    docs_hint: Optional[list[str]] = None,
) -> Optional[tuple[str, dict]]:
    """Kulcsszo-alapu tool valasztas.

    Visszater (tool_name, tool_args) parral, vagy None-nal ha nincs tobb tool.

    Alaplogika:
    - Ha meg soha nem hivtunk list_documents-et -> hivjuk (mindig ezzel kezdunk)
    - Ezutan a kerdesbol detektalt intent alapjan valasztunk
    - Minden tool-t max 1x hivunk, kiveve get_extraction: compare intent
      eseten 2x (a ket doksi adatai kellenek az osszevetes elott)
    """
    tool_names = {t.name for t in bound_tools}

    # Mindig kezdjuk a list_documents-tel ha elerheto es meg nem futott
    if "list_documents" in tool_names and "list_documents" not in previous_tools_called:
        return ("list_documents", {})

    # Intent detektalas
    intent_tool: Optional[str] = None
    for pattern, name, _args in _INTENT_RULES:
        if pattern.search(user_text):
            if name in tool_names:
                intent_tool = name
                break

    if intent_tool is None:
        return None

    is_compare = (intent_tool == "compare_documents")

    # get_extraction hivas-szamlalo -- compare eseten 2 kell, egyebkent 1
    extractions_done = previous_tools_called.count("get_extraction")
    if is_compare and "get_extraction" in tool_names and extractions_done < 2:
        # Keressunk kulonbozo fajlnevekhez extracting-et
        names = _filenames_from_query(user_text, docs_hint)
        if len(names) < 2 and docs_hint and len(docs_hint) >= 2:
            # Fallback: elso ket feltoltott doksi
            names = list(docs_hint)[:2]
        target = names[extractions_done] if extractions_done < len(names) else None
        args = {"filename": target} if target else {}
        return ("get_extraction", args)

    # Loop-vedelem: minden nem-compare intent tool max 1x
    max_calls = {
        "get_extraction": 2 if is_compare else 1,
    }.get(intent_tool, 1)
    if previous_tools_called.count(intent_tool) >= max_calls:
        return None

    # Argumentumok kitoltese
    if intent_tool in {"get_extraction", "validate_document"}:
        names = _filenames_from_query(user_text, docs_hint)
        target = names[0] if names else None
        args = {"filename": target} if target else {}
        return (intent_tool, args)

    if intent_tool == "compare_documents":
        names = _filenames_from_query(user_text, docs_hint)
        if len(names) >= 2:
            return (intent_tool, {"filename_a": names[0], "filename_b": names[1]})
        if docs_hint and len(docs_hint) >= 2:
            return (intent_tool, {"filename_a": docs_hint[0], "filename_b": docs_hint[1]})
        return (intent_tool, {})

    if intent_tool == "search_documents":
        return (intent_tool, {"query": user_text.strip()})

    return (intent_tool, {})


def _synthesize_answer(user_text: str, tool_msgs: List[ToolMessage]) -> str:
    """Tool-output osszegzese magyar valaszba.

    Ez egy egyszeru osszefuzes -- a cel nem a nyelvi finesse, hanem a
    demonstracio: a valasz az agent tool-jainak kimenetere hivatkozik, forras
    jelolessel.
    """
    if not tool_msgs:
        return (
            "Nem tudtam tool-t hivni a kerdesedre. Kerlek, formaldd a kerdest "
            "konkretabban (pl. 'Hany dokumentum van feltoltve?')."
        )

    lines = ["A kerdesedre a kovetkezo toolok eredmenyei alapjan valaszolok:\n"]
    for i, tm in enumerate(tool_msgs, 1):
        tool_name = tm.name if hasattr(tm, "name") and tm.name else f"tool_{i}"
        content = tm.content if isinstance(tm.content, str) else str(tm.content)
        # Rovid osszefoglalas (max 400 karakter per tool-eredmeny)
        summary = content.strip()
        if len(summary) > 400:
            summary = summary[:400] + "..."
        lines.append(f"**{i}. {tool_name}**\n{summary}\n")
    lines.append(
        "\n[Forras: dummy LLM osszegzese -- a valodi valaszhoz Ollama-t (llama3.1:8b) "
        "hasznalj `LLM_PROVIDER=ollama` env valtozoval.]"
    )
    return "\n".join(lines)


class DummyLLM(BaseChatModel):
    """Determinisztikus stub LLM a LangChain BaseChatModel interfeszen.

    Nem vegez valodi nyelvi modellezest -- a tool-valasztas szabaly-alapu
    (_INTENT_RULES), a szintezis pedig a tool-eredmenyek osszefuzese.
    """

    bound_tools: list = Field(default_factory=list)
    docs_hint: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "dummy"

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "DummyLLM":
        return DummyLLM(bound_tools=list(tools), docs_hint=list(self.docs_hint))

    def set_docs_hint(self, docs: list[str]) -> None:
        """Feltoltott fajlok nevei -- a tool-argumentumok kitoltesehez."""
        object.__setattr__(self, "docs_hint", list(docs))

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_text = _last_human_text(messages)
        tool_msgs = _tool_messages(messages)
        previous = [tm.name for tm in tool_msgs if getattr(tm, "name", None)]

        # Tool valasztas
        choice = _choose_tool(user_text, self.bound_tools, previous, self.docs_hint)

        if choice is not None:
            tool_name, tool_args = choice
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            # Nincs tobb tool -> szintezis
            text = _synthesize_answer(user_text, tool_msgs)
            msg = AIMessage(content=text)

        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        # Egyszerusitve: stream = egyetlen generation
        result = self._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            yield gen


def build_dummy_llm() -> DummyLLM:
    return DummyLLM()
