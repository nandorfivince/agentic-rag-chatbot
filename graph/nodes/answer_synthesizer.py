"""Answer synthesizer node -- a vegso valaszt kitolti a state[final_answer]-be.

Standard eset: az agent node utolso AIMessage-e (content mezo) a vegso valasz.
Edge case: ha valami miatt az agent nem adott content-tel valaszt (pl. MAX
ITER limit), akkor a tool_message-ekbol keszitunk egy osszefoglalot.

Ez a node tehat tobbnyire eleg egyszeru -- a valoji munkat az agent node
vegezte, itt csak kinyerjuk a vegredmenyt es a state-be tesszuk.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from graph.state import AgentState
from utils.trace import trace_append


def _last_ai_content(messages) -> str:
    """Utolso AIMessage content-e (content-tel, tool_calls-nal kisebb prioritasu)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _summarize_tool_messages(messages) -> str:
    """Fallback: ha az agent nem adott content-et, a tool-eredmenyeket osszegezzuk."""
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_msgs:
        return "Nem tudtam informaciot szerezni a kerdesedrol."
    lines = ["A tool-ok eredmenyei:"]
    for i, tm in enumerate(tool_msgs, 1):
        name = getattr(tm, "name", f"tool_{i}") or f"tool_{i}"
        content = str(tm.content)
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"  {i}. {name}: {content}")
    return "\n".join(lines)


def answer_synthesizer_node(state: AgentState) -> dict:
    messages = state.get("messages", [])

    final_answer = _last_ai_content(messages)
    if not final_answer.strip():
        final_answer = _summarize_tool_messages(messages)

    trace = trace_append(
        state, "4. answer_synthesizer",
        f"valasz hossz: {len(final_answer)} karakter",
    )
    return {"final_answer": final_answer, "trace": trace}
