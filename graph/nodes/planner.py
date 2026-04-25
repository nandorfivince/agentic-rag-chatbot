"""Planner node -- az intent alapjan javasolt tool-sorrendet ad ki.

A plan-t az agent NEM kotelezoen koveti (a LLM szabadon dontheti melyik
tool-t hivja), de a plan megjelenik a system promptban hintkent. Dummy LLM
eseten a tool-valasztas teljesen kovetni fogja a plan-t (lasd dummy_provider).

A terv altal eloirt "reszfeladatokra bontas" es "autonom dontesheozatal"
kriteriumokat ez a node teljesiti: strukturalt terv + agent dontesi szabadsag.
"""

from __future__ import annotations

from graph.state import AgentState
from utils.trace import trace_append


_PLAN_MAP: dict[str, list[str]] = {
    "list":     ["list_documents"],
    "extract":  ["list_documents", "get_extraction"],
    "search":   ["search_documents"],
    "compare":  ["list_documents", "get_extraction", "get_extraction", "compare_documents"],
    "validate": ["list_documents", "validate_document"],
    "chat":     [],  # Nincs tool, csak direkt valasz
}


def build_plan(intent: str) -> list[str]:
    return list(_PLAN_MAP.get(intent, []))


def planner_node(state: AgentState) -> dict:
    intent = state.get("intent", "chat")
    plan = build_plan(intent)
    trace = trace_append(state, "2. planner", f"plan={plan}")
    return {"plan": plan, "trace": trace}
