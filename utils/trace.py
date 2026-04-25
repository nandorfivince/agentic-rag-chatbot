"""Agent trace helper -- a Streamlit UI sidebar-hoz.

Minden LangGraph node beirhatja az aktualis lepest a state["trace"] listaba.
A UI-ban ezt vissza lehet olvasni, es kirajzolni "mit csinalt az agent"
lepes-szinten (intent -> plan -> tool_calls -> synth -> validator).
"""

from __future__ import annotations

import time
from typing import Any


def now_prefix() -> str:
    return time.strftime("%H:%M:%S")


def trace_append(state: dict, step: str, meta: Any = None) -> list[str]:
    """Bovitjuk a trace listat egy uj bejegyzessel, es vissza is adjuk.

    A LangGraph immutable state-pattern szerint mindig uj listat adunk vissza
    (nem mutaljuk az eredetit) -- igy a checkpoint-ing es a branching hibatlan.
    """
    existing = list(state.get("trace", []))
    entry = f"[{now_prefix()}] {step}"
    if meta is not None:
        entry += f"  {meta}"
    existing.append(entry)
    return existing


def trace_format(trace: list[str]) -> str:
    return "\n".join(trace)
