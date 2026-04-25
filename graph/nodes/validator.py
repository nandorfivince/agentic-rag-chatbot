"""Validator node -- anti-hallucinacio es forras-hivatkozas ellenorzes.

Mit ellenoriz?
1. Ha volt tool-hivas (ToolMessage a messages-ben), a vegso valasznak
   hivatkoznia kell a forrasokra (fajlnev szerepelj benne).
2. Ha a valasz nagyon rovid (< 20 karakter) es voltak tool-eredmenyek,
   gyanus: valoszinuleg az agent nem valaszolt a kerdesre.

Mi tortenik hiba eseten?
- A validator_retry_count novel es HumanMessage-t injektalunk a messages-be
  ami visszakuldi az agent-et a loop-ba.
- Max MAX_VALIDATOR_RETRIES (ket) probalkozas utan elfogadja az eredmenyt.

Kilepesi kapu: a conditional edge a main_graph-ban dont, hogy
(VALIDATOR -> END) vagy (VALIDATOR -> AGENT) tortenjen.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage

from config import settings
from graph.state import AgentState
from utils.trace import trace_append


def _has_source_reference(answer: str, tool_messages: list[ToolMessage]) -> bool:
    """Van-e forras-hivatkozas a valaszban?

    Elfogadott formak:
    - [Forras: fajlnev.pdf]
    - "forras:" kulcsszo
    - bármilyen fajlnev amelyet a tool eredmeny tartalmazott, szerepel-e a
      valaszban (pl. "szamla_januar.pdf" mention)
    """
    if "[Forras:" in answer or "[Forrás:" in answer:
        return True
    low = answer.lower()
    if "forras:" in low or "forrás:" in low:
        return True
    # Keressuk a fajlneveket amik a tool-eredmenyekben voltak
    for tm in tool_messages:
        content = str(tm.content).lower()
        # Egyszeru heurisztika: keressunk "*.pdf" vagy "*.docx" mintat a tool-output-ban
        import re
        for m in re.finditer(r"([\w\-]+\.(?:pdf|docx))", content):
            if m.group(1).lower() in low:
                return True
    return False


def validator_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    answer = state.get("final_answer", "")
    retry_count = state.get("validator_retry_count", 0)

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    # Ha nem volt tool-hivas (pl. intent="chat"), nincs mit ellenoriznunk
    if not tool_messages:
        trace = trace_append(
            state, "5. validator",
            "OK (nem voltak tool-hivasok, chat mod)",
        )
        return {"trace": trace}

    has_source = _has_source_reference(answer, tool_messages)
    too_short = len(answer.strip()) < 20

    if (not has_source or too_short) and retry_count < settings.max_validator_retries:
        # Visszakuldjuk az agent-hez egy explicit utasitassal
        reason = []
        if not has_source:
            reason.append("nincs forras-hivatkozas")
        if too_short:
            reason.append("tul rovid valasz")
        trace = trace_append(
            state, "5. validator",
            f"FAIL ({', '.join(reason)}), retry {retry_count + 1}/{settings.max_validator_retries}",
        )
        retry_msg = HumanMessage(
            content=(
                "A valasz nem elfogadhato: "
                f"{', '.join(reason)}. "
                "Kerlek hivd meg ujra a szukseges tool-okat, es a "
                "vegso valaszban hivatkozz a forrasra [Forras: fajlnev.pdf] "
                "formatumban. Legalabb 20 karakter hosszu legyen."
            )
        )
        return {
            "messages": [retry_msg],
            "validator_retry_count": retry_count + 1,
            "trace": trace,
        }

    # OK (vagy elfogytak a retry-k, elfogadjuk)
    status = "OK" if (has_source and not too_short) else "elfogadva (max retry)"
    trace = trace_append(state, "5. validator", status)
    return {"trace": trace}


def should_retry(state: AgentState) -> str:
    """Conditional edge helper: a validator node utan dont hogy END vagy agent.

    Az agent node utan ujra futna a tool-executor loop, majd a synthesizer,
    majd ide -- ez a standard "validate-then-retry" pattern.
    """
    retry_count = state.get("validator_retry_count", 0)
    # Ha a validator futast kovetoen a retry-szamlalo novekedett, agent-hez
    # megyunk. Kulonben END.
    messages = state.get("messages", [])
    # A validator_node csak akkor ad vissza messages-t, ha a retry-t kezdi
    # A LangGraph state merge utan ha az utolso uzenet HumanMessage (a validator-tol),
    # akkor visszamegyunk az agenthez.
    if retry_count > 0 and messages:
        from langchain_core.messages import HumanMessage
        if isinstance(messages[-1], HumanMessage):
            # Csak akkor terit vissza agent-hez, ha ez a LEGUTOLSO uzenet
            # a validator retry-je (tehat meg nem dolgozta fel az agent)
            return "agent"
    return "end"
