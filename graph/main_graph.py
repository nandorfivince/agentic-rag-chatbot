"""Main LangGraph workflow -- a fo agentic pipeline osszealitasa.

Topologia:

    START
     |
     v
    intent_classifier   (keyword alapu, gyors)
     |
     v
    planner             (intent -> tool-sorrend hint)
     |
     v
    agent  <-----------+
     |                 |
     | (tools_cond)    |
     +---> tool_node --+
     |
     v  (no tool_call)
    answer_synthesizer
     |
     v
    validator
     |
     | (should_retry)
     +---> agent (retry, max 2x)
     |
     v
    END

Ez 5 fo node (intent/planner/agent/synth/validator) + 1 prebuilt (ToolNode)
= 6 node osszesen, a plan es a feladatleiras kriteriumat bőven teljesiti.

A RAG subgraph (graph/rag_subgraph.py) NEM szamit ide -- a search_documents
tool invokalja kozvetlenul, ahogy a feladatleiras megengedi: "dedikalt,
modularis RAG algrafot (subgraph), amely a fo workflow-bol hivhato, de nem
szamit bele a 3-5 csomópontba."
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.nodes.agent import build_agent_node
from graph.nodes.answer_synthesizer import answer_synthesizer_node
from graph.nodes.intent_classifier import intent_classifier_node
from graph.nodes.planner import planner_node
from graph.nodes.validator import should_retry, validator_node
from graph.state import AgentState


def build_main_graph(llm, tools, with_checkpointer: bool = True) -> Any:
    """A main agentic LangGraph compile-olasa.

    Args:
        llm: BaseChatModel (Ollama vagy Dummy). A bind_tools()-szal kapcsolodik.
        tools: list[BaseTool] a tools/ modulbol (build_tools(context) eredmeny).
        with_checkpointer: True eseten MemorySaver -- a follow-up kerdesekhez
            a beszelgetes-memoria perzisztal thread_id szintjen.

    Returns:
        Compiled graph, amit a `graph.invoke(state, config={"configurable":
        {"thread_id": "..."}})` mintaval lehet futtatni.
    """
    llm_with_tools = llm.bind_tools(tools)
    agent_node = build_agent_node(llm_with_tools)

    graph = StateGraph(AgentState)

    # 6 node
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_node", ToolNode(tools))
    graph.add_node("answer_synthesizer", answer_synthesizer_node)
    graph.add_node("validator", validator_node)

    # Linearis szakasz az elso 3 lepeshez
    graph.add_edge(START, "intent_classifier")
    graph.add_edge("intent_classifier", "planner")
    graph.add_edge("planner", "agent")

    # React loop: agent -> tool_node (ha van tool_call) -> agent
    # vagy agent -> answer_synthesizer (ha csak content-tel valaszolt).
    # A tools_condition prebuilt helper az utolso AIMessage.tool_calls mezo
    # alapjan dont.
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tool_node",
            END: "answer_synthesizer",
        },
    )
    graph.add_edge("tool_node", "agent")

    # Synth -> validator linearisan
    graph.add_edge("answer_synthesizer", "validator")

    # Validator conditional: (a) retry -> vissza agent, (b) end -> END
    graph.add_conditional_edges(
        "validator",
        should_retry,
        {
            "agent": "agent",
            "end": END,
        },
    )

    checkpointer = MemorySaver() if with_checkpointer else None
    return graph.compile(checkpointer=checkpointer)


def run_once(graph, user_question: str, thread_id: str = "default") -> dict:
    """Egyszeri kerdes-valasz ciklus futtatasa a graph-on.

    Ezzel konnyeb tesztelni/evaluate-elni. Az app.py a streaming variansot
    hasznalja, de erre lasd a graph.stream(...) hivast.

    Returns: {"answer", "trace", "messages", "iteration_count"}.
    """
    from langchain_core.messages import HumanMessage

    initial: AgentState = {
        "messages": [HumanMessage(content=user_question)],
        "iteration_count": 0,
        "validator_retry_count": 0,
        "trace": [],
    }
    config = {"configurable": {"thread_id": thread_id}}
    final = graph.invoke(initial, config=config)

    return {
        "answer": final.get("final_answer", ""),
        "trace": final.get("trace", []),
        "messages": final.get("messages", []),
        "iteration_count": final.get("iteration_count", 0),
    }
