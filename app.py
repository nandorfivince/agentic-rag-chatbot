"""Streamlit UI -- Agentic RAG Chatbot a magyar uzleti dokumentum intelligence domainra.

Layout:

    +-------------------------------------------------------------------+
    |  Agentic RAG Chatbot - Document Intelligence                      |
    +-------------------------------+-----------------------------------+
    |  Dokumentumok feltoltese      |  Agent lepesek (trace)            |
    |  [drag & drop box]            |  [1] intent_classifier: compare   |
    |  * szamla_januar.pdf (OK)     |  [2] planner: [list, get, ...]    |
    |  * szamla_marcius.pdf (OK)    |  [3] agent iter=1: tool_calls=1   |
    |                               |  [4] tool_node: list_documents    |
    |  Chat                         |  [5] agent iter=2: tool_calls=1   |
    |  User: mennyivel dragabb?     |  [6] tool_node: get_extraction    |
    |  Asst: 25%-kal [Forras: ...]  |  [7] agent iter=3: tool_calls=0   |
    |  [input]                      |  [8] answer_synth: 142 char       |
    |                               |  [9] validator: OK                |
    +-------------------------------+-----------------------------------+

Session state:
  - provider: BaseChatModel (Ollama vagy Dummy)
  - store: HybridStore (ChromaDB + BM25)
  - tool_context: ToolContext (docs, extractions, trace_hook)
  - tools: list[BaseTool]
  - graph: compiled LangGraph
  - chat_history: list[dict] {role, content}
  - last_trace: list[str]
  - uploaded_files_hash: set[str] (ne tolsuk fel ugyanazt ketszer)
  - thread_id: UUID -- checkpoint kontinuitas
"""

from __future__ import annotations

import uuid
from io import BytesIO

import streamlit as st
from langchain_core.messages import HumanMessage

from config import settings
from graph import build_main_graph
from ingest import chunk_document, embed_batch, load_bytes
from llm import get_llm
from store import HybridStore
from tools import ToolContext, build_tools


st.set_page_config(
    page_title="Agentic RAG Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Session init -- csak egyszer futjuk le session-onkent
# ---------------------------------------------------------------------------

def _init_session():
    if "provider" in st.session_state:
        return

    with st.spinner("LLM es vektortar inicializalasa..."):
        st.session_state.provider = get_llm()
        st.session_state.store = HybridStore()
        st.session_state.tool_context = ToolContext(
            store=st.session_state.store,
            documents={},
            extractions={},
        )
        st.session_state.tools = build_tools(st.session_state.tool_context)
        st.session_state.graph = build_main_graph(
            st.session_state.provider,
            st.session_state.tools,
            with_checkpointer=True,
        )
        st.session_state.chat_history = []
        st.session_state.last_trace = []
        st.session_state.uploaded_files_hash = set()
        st.session_state.thread_id = str(uuid.uuid4())

    # Dummy LLM eseten a docs_hint mezot kitoltjuk (feltoltott fajlok) -- a
    # provider ezekbol valasztja a tool-argumentumot, ha a kerdes nem specifikal.
    _sync_docs_hint()


def _sync_docs_hint():
    """Dummy LLM docs_hint frissitese a feltoltott fajl-lista alapjan."""
    provider = st.session_state.get("provider")
    if provider is not None and hasattr(provider, "set_docs_hint"):
        docs = list(st.session_state.tool_context.documents.keys())
        provider.set_docs_hint(docs)


def _ingest_file(file_name: str, content: bytes) -> tuple[bool, str]:
    """Egy feltoltott fajl feldolgozasa: load + chunk + embed + store + tool_context."""
    try:
        doc = load_bytes(file_name, content)
    except Exception as e:
        return False, f"Betoltesi hiba: {e}"

    if not doc.full_text.strip():
        return False, "A fajlbol nem sikerult szoveget kinyerni."

    chunks = chunk_document(doc)
    if chunks:
        texts = [c["text"] for c in chunks]
        try:
            embeddings = embed_batch(texts)
        except Exception as e:
            return False, f"Embedding hiba: {e}"
        st.session_state.store.add_chunks(chunks, embeddings)

    st.session_state.tool_context.documents[file_name] = doc
    # Extraction cache torles (az uj fajl utan majd on-demand generalodik)
    st.session_state.tool_context.extractions.pop(file_name, None)
    _sync_docs_hint()
    return True, f"Feldolgozva: {doc.file_name} ({len(chunks)} chunk)"


# ---------------------------------------------------------------------------
# UI reszek
# ---------------------------------------------------------------------------

def _render_header():
    st.title("Agentic RAG Chatbot — Document Intelligence")
    st.caption(
        f"LLM provider: **{settings.llm_provider}** · "
        f"Embedding: `{settings.embedding_model}` · "
        f"Vektortar: ChromaDB + BM25 hibrid"
    )


def _render_upload(col):
    with col:
        st.subheader("Dokumentumok feltoltese")
        uploaded = st.file_uploader(
            "PDF, DOCX, PNG, JPG, TXT",
            type=["pdf", "docx", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            for upfile in uploaded:
                content = upfile.getvalue()
                file_hash = f"{upfile.name}:{len(content)}"
                if file_hash in st.session_state.uploaded_files_hash:
                    continue
                with st.spinner(f"Feldolgozas: {upfile.name}"):
                    ok, msg = _ingest_file(upfile.name, content)
                if ok:
                    st.session_state.uploaded_files_hash.add(file_hash)
                    st.success(msg)
                else:
                    st.error(msg)

        docs = st.session_state.tool_context.documents
        if docs:
            st.markdown(f"**Feltoltva: {len(docs)} dokumentum**")
            for name in sorted(docs):
                st.markdown(f"  - `{name}`")
        else:
            st.info("Meg nincs feltoltott dokumentum. A chat mukodik RAG nelkul is.")


def _render_chat(col):
    with col:
        st.subheader("Chat")
        # Chat history kirajzolas
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        prompt = st.chat_input("Kerdezd meg a dokumentumaidat...")
        if prompt:
            # Append -> rerun -- a kovetkezo rendernel a history loop rajzolja
            # ki az uj user-uzenetet, igy nincs dupla megjelenites.
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Agent dolgozik..."):
                answer, trace = _invoke_graph(prompt)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.session_state.last_trace = trace
            st.rerun()


def _invoke_graph(question: str) -> tuple[str, list[str]]:
    """A graph futtatasa egy kerdesre. Hibaturo."""
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial = {
            "messages": [HumanMessage(content=question)],
            "iteration_count": 0,
            "validator_retry_count": 0,
            "trace": [],
        }
        final = st.session_state.graph.invoke(initial, config=config)
        answer = final.get("final_answer") or "(ures valasz)"
        trace = final.get("trace", [])
        return answer, trace
    except Exception as e:
        return f"Hiba: {e}", [f"exception: {e}"]


def _render_trace(col):
    with col:
        st.subheader("Agent lepesek")
        trace = st.session_state.get("last_trace", [])
        if not trace:
            st.info("Tegyel fel egy kerdest a chat-ben -- itt fog lathatni "
                    "az agent lepeseit (intent, plan, tool-hivasok, validator).")
            return
        for line in trace:
            st.text(line)

        st.markdown("---")
        st.subheader("Session info")
        st.text(f"thread_id: {st.session_state.thread_id[:8]}...")
        st.text(f"store chunks: {st.session_state.store.count()}")
        st.text(f"doksi: {len(st.session_state.tool_context.documents)}")
        st.text(f"extraction cache: {len(st.session_state.tool_context.extractions)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_session()
    _render_header()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        _render_upload(col_left)
        st.markdown("---")
        _render_chat(col_left)

    _render_trace(col_right)


if __name__ == "__main__":
    main()
