"""Agent node -- az LLM + bind_tools kapcsolata.

Ez a React-pattern magja: az LLM megkapja a messages listat + a bind_tools-on
keresztul elerheto tool-regisztratiot, es dont, hogy:
  - tool-t akar hivni (AIMessage.tool_calls nem ures),
  - vagy adott a vegleges szoveges valasz (content kitoltve).

A tool_node (kulon LangGraph node) ezt a dontest hajtja vegre, es
ToolMessage-ekkel terit vissza. A loop akkor all le, amikor az agent
content-tel valaszol (nincs uj tool_call).

Vegtelen-loop vedelem: az iteration_count counter minden hivasnal nő.
Ha elerte a MAX_ITERATIONS limitet, kenyszeriben vegleges valaszt ad vissza.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, SystemMessage

from config import settings
from graph.state import AgentState
from utils.trace import trace_append


# A rendszer-prompt -- az agent LLM ezt latja kontextuskent. Magyarul tervezett,
# explicit szabalyokkal hogy a tool-hivas akkor is megbizhato legyen, ha a
# modell (mondjuk llama3.1:8b) nem a legmodernebb.
SYSTEM_PROMPT = """Egy Document Intelligence asszisztens vagy.
A felhasznalo kerdeseire a megadott tool-okkal valaszolj.

SZABALYOK:
1. MINDIG hivj legalabb 1 tool-t mielott valaszolnal (soha ne talalgass).
2. Ha listazni kell, hasznald a list_documents tool-t.
3. Ha adatot keresel egy specifikus doksirol (osszeg, datum, tetel),
   hasznald a get_extraction tool-t a filename-nel.
4. Ha hibat vagy matek-ellenorzest kernek, hasznald a validate_document tool-t.
5. Ha ket doksit kell osszevetni, hasznald a compare_documents tool-t.
6. Ha szovegben kell keresni (klauzula, datum, stb.), hasznald a search_documents tool-t.
7. A vegso valaszban MINDIG hivatkozz a forrasra: [Forras: filename.pdf].
8. Magyar nyelven, tomoren valaszolj. Ne talalj ki szamot vagy datumot -- csak a tool kimenetebol dolgozz.

A KOVETKEZO TERV javaslat lett keszitve a kerdesedhez (irani tud, de szabadon ternhetsz el):
{plan_hint}
"""


def build_agent_node(llm_with_tools) -> Callable[[AgentState], dict]:
    """Visszater egy agent node functiont, amiben az LLM be van kotve.

    A closure-ben tartjuk az llm_with_tools-t, mert az az app.py session-ban
    buildolodik (provider + bind_tools(tools)).
    """

    def agent_node(state: AgentState) -> dict:
        iter_count = state.get("iteration_count", 0)

        if iter_count >= settings.max_iterations:
            trace = trace_append(
                state, "3. agent",
                f"MAX ITER ({settings.max_iterations}) -- kenyszeriben zaras",
            )
            return {
                "messages": [AIMessage(
                    content=(
                        f"Elertem a maximalis iteraciot ({settings.max_iterations}). "
                        "Kerlek, formaldd at a kerdest tomorebben."
                    ),
                )],
                "iteration_count": iter_count + 1,
                "trace": trace,
            }

        # A system prompt az uzenet-lista elejen -- a tervet a plannerbol rakjuk be
        plan = state.get("plan", [])
        plan_hint = ", ".join(plan) if plan else "(nincs eloirt terv, dontsd el magad)"
        system_msg = SystemMessage(
            content=SYSTEM_PROMPT.format(plan_hint=plan_hint)
        )

        messages = [system_msg] + list(state.get("messages", []))

        # Dummy LLM: kell a docs_hint a tool-argumentumok kitoltesehez
        if hasattr(llm_with_tools, "set_docs_hint"):
            # Ha a rendszer (app.py) erre feliratkozott, beallitja session-onkent.
            # Most ez a resz a dummy provider-hez szabott lasd dummy_provider.py.
            pass

        response = llm_with_tools.invoke(messages)

        tool_call_count = (
            len(response.tool_calls) if hasattr(response, "tool_calls")
            and response.tool_calls else 0
        )
        trace = trace_append(
            state, "3. agent",
            f"iter={iter_count + 1}, tool_calls={tool_call_count}, "
            f"content_len={len(response.content) if response.content else 0}",
        )

        return {
            "messages": [response],
            "iteration_count": iter_count + 1,
            "trace": trace,
        }

    return agent_node
