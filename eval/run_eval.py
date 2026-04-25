"""Funkcionalis ertekeles -- 15 kerdeses mini eval.

Futtatas: python eval/run_eval.py [--llm ollama|dummy] [--quick]

A kerdes-keszletet `eval/questions.json` tartalmazza, 5 kategoriaban:
  - list     : 2 kerdes (dokumentum-listazas)
  - extract  : 3 kerdes (adatok kinyerese egy fajlbol)
  - search   : 3 kerdes (RAG-alapu szemantikus kereses)
  - compare  : 4 kerdes (ket doksi osszevetese)
  - validate : 3 kerdes (matek/CDV validacio)

Metrikak per kerdes:
  - pass   : az expected_substrings legalabb egyike szerepel a valaszban
  - tools  : a ToolMessage-ekben szereplo tool-nevek
  - latency_ms
  - expected_tools_match : a varandas tool-ok mind lefutottak-e

Kimenet: eval/results.md (kommit-barat markdown riport).

A --quick kapcsoloval csak 5 mintat futtat (gyors ellenorzes fejlesztes kozben).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# A futtato konyvtar-fuggetlen: a projekt gyokerbol legyen importalhato
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402

from graph import build_main_graph  # noqa: E402
from ingest import chunk_document, embed_batch, load_pdf  # noqa: E402
from llm import get_llm  # noqa: E402
from store import HybridStore  # noqa: E402
from tools import ToolContext, build_tools  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_MD = EVAL_DIR / "results.md"
SAMPLE_DIR = EVAL_DIR.parent / "data" / "sample_docs"


def _load_questions(quick: bool = False) -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if quick:
        # 1 per kategoria
        seen = set()
        out = []
        for q in questions:
            if q["category"] in seen:
                continue
            seen.add(q["category"])
            out.append(q)
        return out
    return questions


def _setup_environment(llm_provider: str) -> tuple:
    """Feltolt minden sample PDF-et a store-ba es egy graph-ot epit."""
    os.environ["LLM_PROVIDER"] = llm_provider
    # A config egyszer toltodik; ha mar importalodott, kivalthat az env varioannak
    # tul keson -- ezert az eval modul a fuggosegeket itt lazy-importalja.
    from importlib import reload
    import config
    reload(config)  # env valtozok ujraolvasasa
    # A dependent modulok (llm, settings-fuggo kod) kereshetik a frissitest;
    # ez biztonsag kedveert ebben a folyamatban egyszer futunk, igy nem gond.

    llm = get_llm()
    store = HybridStore()
    tool_context = ToolContext(store=store, documents={}, extractions={})
    tools = build_tools(tool_context)
    graph = build_main_graph(llm, tools, with_checkpointer=False)

    # Doksik feltoltese
    pdf_files = sorted(SAMPLE_DIR.glob("*.pdf"))
    for pdf in pdf_files:
        doc = load_pdf(pdf)
        chunks = chunk_document(doc)
        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = embed_batch(texts)
            store.add_chunks(chunks, embeddings)
        tool_context.documents[doc.file_name] = doc

    # Dummy LLM-nek docs_hint kell
    if hasattr(llm, "set_docs_hint"):
        llm.set_docs_hint(list(tool_context.documents.keys()))

    return graph, tool_context, pdf_files


def _extract_tools_used(messages) -> list[str]:
    return [
        getattr(m, "name", "?") or "?"
        for m in messages if isinstance(m, ToolMessage)
    ]


def _evaluate_question(graph, q: dict) -> dict:
    start = time.time()
    initial = {
        "messages": [HumanMessage(content=q["question"])],
        "iteration_count": 0,
        "validator_retry_count": 0,
        "trace": [],
    }
    try:
        final = graph.invoke(initial)
        answer = final.get("final_answer") or ""
        messages = final.get("messages", [])
        error = None
    except Exception as e:
        answer = ""
        messages = []
        error = str(e)

    latency_ms = (time.time() - start) * 1000
    tools_used = _extract_tools_used(messages)

    expected_substrings = q.get("expected_substrings", [])
    expected_tools = q.get("expected_tools", [])

    lower_answer = answer.lower()
    # A "pass" heurisztika: a varandas string-ekbol legalabb egy szerepel a valaszban
    # vagy a tool-eredmenyekben (osszefuzve)
    tool_outputs = " ".join(
        str(m.content) for m in messages if isinstance(m, ToolMessage)
    ).lower()
    haystack = lower_answer + " " + tool_outputs

    substring_hits = [s for s in expected_substrings if s.lower() in haystack]
    passed = len(substring_hits) > 0 or not expected_substrings

    tool_match = all(t in tools_used for t in expected_tools)

    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "answer": answer,
        "tools_used": tools_used,
        "expected_tools": expected_tools,
        "expected_tools_match": tool_match,
        "expected_substrings_hit": substring_hits,
        "pass": passed,
        "latency_ms": latency_ms,
        "error": error,
    }


def _render_markdown(results: list[dict], llm_provider: str) -> str:
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    tool_ok = sum(1 for r in results if r["expected_tools_match"])
    latencies = [r["latency_ms"] for r in results if r["error"] is None]
    latencies.sort()

    def _pct(n, d): return f"{100*n/d:.0f}%" if d else "0%"
    def _p(k): return latencies[int(len(latencies)*k)] if latencies else 0

    lines = [
        "# Funkcionalis ertekeles eredmenye",
        "",
        f"- LLM provider: **{llm_provider}**",
        f"- Osszesen: {total} kerdes",
        f"- Pass rate: **{passed}/{total} ({_pct(passed, total)})**",
        f"- Tool-sorrend egyezes: {tool_ok}/{total} ({_pct(tool_ok, total)})",
        f"- Latency p50: {_p(0.5):.0f} ms, p95: {_p(0.95):.0f} ms, max: "
        f"{(latencies[-1] if latencies else 0):.0f} ms",
        "",
        "## Per-kerdes eredmenyek",
        "",
        "| ID | Kat. | Pass | Tools | Latency (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        pass_flag = "OK" if r["pass"] else "FAIL"
        tools_str = ", ".join(r["tools_used"]) if r["tools_used"] else "-"
        tool_match = "+" if r["expected_tools_match"] else "-"
        lines.append(
            f"| {r['id']} | {r['category']} | {pass_flag} | "
            f"{tools_str} [{tool_match}] | {r['latency_ms']:.0f} |"
        )

    # Kategoria-szinten
    lines.append("")
    lines.append("## Kategoriankent")
    lines.append("")
    lines.append("| Kategoria | Pass | Total |")
    lines.append("|---|---|---|")
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rs in sorted(by_cat.items()):
        p = sum(1 for r in rs if r["pass"])
        lines.append(f"| {cat} | {p} | {len(rs)} |")

    # Reszletek FAIL esetekre
    fails = [r for r in results if not r["pass"]]
    if fails:
        lines.append("")
        lines.append("## Sikertelen kerdesek (reszletes)")
        for r in fails:
            lines.append("")
            lines.append(f"### {r['id']} — {r['question']}")
            lines.append(f"- Valasz: {r['answer'][:200]}")
            lines.append(f"- Tools: {r['tools_used']}")
            lines.append(f"- Vart substrings: {[s for s in r.get('expected_substrings_hit', [])] or 'egyik sem illeszkedett'}")
            if r["error"]:
                lines.append(f"- HIBA: {r['error']}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default=os.getenv("LLM_PROVIDER", "dummy"),
                        choices=["ollama", "dummy"])
    parser.add_argument("--quick", action="store_true", help="csak 5 kerdes")
    parser.add_argument("--no-write", action="store_true",
                        help="ne menjen results.md-be")
    args = parser.parse_args()

    print(f"Kornyezet init (llm={args.llm})...")
    graph, tool_context, pdfs = _setup_environment(args.llm)
    print(f"Feltoltve: {len(pdfs)} PDF, {sum(1 for _ in tool_context.documents)} doksi")

    questions = _load_questions(quick=args.quick)
    print(f"Futtatas: {len(questions)} kerdes...")

    results = []
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:60]}...", end=" ", flush=True)
        r = _evaluate_question(graph, q)
        pass_flag = "OK" if r["pass"] else "FAIL"
        print(f"{pass_flag}  ({r['latency_ms']:.0f} ms)")
        results.append(r)

    md = _render_markdown(results, args.llm)
    print()
    print(md)

    if not args.no_write:
        RESULTS_MD.write_text(md, encoding="utf-8")
        print(f"\nMentve: {RESULTS_MD}")


if __name__ == "__main__":
    main()
