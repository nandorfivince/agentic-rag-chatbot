"""LLM factory -- a config.py LLM_PROVIDER beallitasa alapjan ad vissza peldanyt.

Hasznalata:

    from llm import get_llm
    llm = get_llm()
    llm_with_tools = llm.bind_tools(my_tools)
    response = llm_with_tools.invoke([HumanMessage(content="...")])

Interfesz: langchain_core.language_models.BaseChatModel (Ollama es Dummy is
ezt implementalja -- igy a LangGraph workflow nem tudja, melyik fut).
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from config import settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """A config.llm_provider alapjan Ollama vagy Dummy chat modellt ad vissza."""
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from llm.ollama_provider import build_ollama_llm
        return build_ollama_llm(temperature=temperature)

    if provider == "dummy":
        from llm.dummy_provider import build_dummy_llm
        return build_dummy_llm()

    raise ValueError(
        f"Ismeretlen LLM_PROVIDER: '{settings.llm_provider}'. "
        "Elfogadott ertekek: 'ollama', 'dummy'."
    )


__all__ = ["get_llm"]
