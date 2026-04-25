"""Szoveg chunkolas termeszetes vagasi pontokon.

Strategia (rekurziv):
1. Szet-vagas bekezdesekre (\n\n)
2. Ha egy bekezdes > chunk_size, mondat-alapu (. / ! / ?)
3. Ha egy mondat is > chunk_size, ha-d-cut karakterhataron
4. Kis chunkok osszevonasa chunk_size alatt
5. Overlap hozzaadasa (elozo chunk utolso N karakter)

A cel: az embedding es a kereses minosege -- egy chunk egy egysegnyi
szemantikus tartalom legyen (bekezdes vagy par mondat), ne felig vagott szoveg.
"""

from __future__ import annotations

import re
from typing import Optional

from config import settings


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    # 2+ ujsor vagy 1 ujsor + tabs/spaces -> bekezdes-hatar
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(paragraph)
    return [s.strip() for s in sentences if s.strip()]


def _hard_cut(text: str, max_chars: int) -> list[str]:
    """Utolso menedek: karakter-hatar cut ha egy mondat is tul hosszu."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[str]:
    """Szoveg szetvagasa chunk_size meretu, chunk_overlap atfedessu reszekre."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    # 1. Bekezdesek
    units: list[str] = []
    for para in _split_paragraphs(text):
        if len(para) <= size:
            units.append(para)
            continue
        # 2. Mondatok
        for sent in _split_sentences(para):
            if len(sent) <= size:
                units.append(sent)
            else:
                # 3. Hard cut
                units.extend(_hard_cut(sent, size))

    # 4. Osszevonas size alatt
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= size:
            current = current + "\n\n" + unit
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    # 5. Overlap -- az elozo chunk utolso `overlap` karakteret hozzafuzzuk
    if overlap > 0 and len(chunks) > 1:
        with_overlap: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            with_overlap.append(prev_tail + "\n\n" + chunks[i])
        chunks = with_overlap

    return chunks


def chunk_document(doc, chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> list[dict]:
    """Document -> chunk dict-ek forrasmezovel.

    Visszater:
        [{"text": "...", "metadata": {"source": file_name, "chunk_index": i,
                                       "page": 1}}, ...]

    A page mezo egyelore 1 -- page-szintu chunking a jovoben bovitheto (most a
    full_text-et bontjuk egyszerusen).
    """
    texts = chunk_text(doc.full_text, chunk_size, chunk_overlap)
    return [
        {
            "text": t,
            "metadata": {
                "source": doc.file_name,
                "chunk_index": i,
                "page": 1,
            },
        }
        for i, t in enumerate(texts)
    ]
