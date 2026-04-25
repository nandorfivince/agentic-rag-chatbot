"""AgentState -- a main LangGraph allapot-struktura.

Minden node ezt a TypedDict-et olvassa/modositja. A LangGraph
automatikusan merge-eli a visszaadott reszlet-modositasokat a state-be.

A `messages` mezohoz az `add_messages` reducer tartozik -- ez appendeli az
uzeneteket, nem felulirja (a React agent patternhez ez a standard).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Beszelgetes uzenetei (user + assistant + tool messages)
    messages: Annotated[list[BaseMessage], add_messages]

    # Intent detektalas kimenete (intent_classifier node)
    # Ertekei: "list" | "extract" | "search" | "compare" | "validate" | "chat"
    intent: str

    # Planner kimenete -- a javasolt tool-sorrend (hint az agent node-nak)
    plan: list[str]

    # Agent iteracios szamlalo (vegtelen loop vedelem)
    iteration_count: int

    # Validator visszadobas szamlalo (max retry)
    validator_retry_count: int

    # Vegso valasz a felhasznalonak (answer_synthesizer kitolti)
    final_answer: str

    # Trace: UI sidebar-on latszo lepes-log
    trace: list[str]
