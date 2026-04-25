"""Ollama provider -- ChatOllama wrapper.

A langchain-ollama csomag ChatOllama-ja mar teljes BaseChatModel, tamogatja a
bind_tools(), invoke(), stream() metodusokat. Itt csak a config-bol parametert
olvasunk es egy factoryt adunk.

Tool-use: a Llama 3.1 natively tamogatja a function calling-ot (Ollama
0.3.0-tol). A ChatOllama automatikusan konvertalja a bind_tools()-nal atadott
LangChain tool-okat Ollama formatumba.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from config import settings


def build_ollama_llm(temperature: float = 0.0) -> ChatOllama:
    """Ollama chat modell peldanyositasa a config alapjan.

    temperature=0 alapertek -- determinisztikus valaszok a reprodukalhato
    ertekeleshez es load teszthez.
    """
    return ChatOllama(
        base_url=settings.ollama_host,
        model=settings.ollama_model,
        temperature=temperature,
    )
