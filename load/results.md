# Load test eredmenye

- LLM provider: **dummy**
- Osszes query: 100
- Sikeres: 100/100 (100.0%)
- Teljes falido: 0.58 sec
- **Atbocsatokepesseg: 172.7 query/sec**
- Kornyezet setup: 5760 ms (8 PDF feltoltes + embedding indexeles)
- Warm-up query: 9 ms (sentence-transformers modell betoltes)

## Latency eloszlas (ms) -- a 100 mereshez

| Statisztika | Ertek (ms) |
|---|---|
| Min | 1.9 |
| p50 (median) | 3.5 |
| Atlag | 5.8 |
| p95 | 17.9 |
| p99 | 24.6 |
| Max | 24.6 |
| Stdev | 5.2 |

## Per-intent latency atlag

| Intent | Count | Atlag (ms) | p95 (ms) |
|---|---|---|---|
| list | 14 | 2.1 | 2.3 |
| extract | 18 | 3.1 | 3.4 |
| search | 14 | 17.8 | 24.6 |
| compare | 29 | 5.5 | 6.5 |
| validate | 16 | 3.4 | 5.1 |
| chat | 9 | 3.5 | 4.7 |

## Bottleneck elemzes

A szekvenciális 100 futtatas alapjan a fő szűk keresztmetszetek:

1. **Embedding warm-up**: az első `search_documents` hivas `9 ms` (sentence-transformers modell lazy-load). Az `lru_cache`-elt utani hivasok > 50x gyorsabbak (<50 ms).

2. **RAG kereses (search intent)**: atlag `18 ms`, mig a tobbi intent (list/extract/compare/validate) atlag `4 ms`. A kulonbseg oka: a query embedding (egy SentenceTransformer forward pass) + Chroma `cosine_distance` lekerdezes + BM25 rangsor osszefuzes RRF-fel.

3. **ChromaDB + BM25 hibrid search**: a BM25 in-memory, minden `add_chunks()` utan ujraepul. 8 dokumentum ~40 chunk eseten elhanyagolhato (<5 ms), de 100+ dokumentum eseten skalazas elottet ez is bottleneck lehet.

## Optimalizalasi javaslatok

### 1. Sentence-transformers modell pre-loading startup-kor

A jelenleg az `embedder.embed()` lusta-tolti a modellt, igy az elso query 9 ms kozott ott tart. Megoldas: az `app.py` session init soranban egy `_ = embed('warmup')` hivas a spinnerrel egyidoben. Igy az elso tenyleges user-query mar gyors. **Varhato nyeresseg**: p99 latency 30-40%-os csokkenes.

### 2. RAG search `top_k` csokkentese, vagy Chroma `hnsw` parameter finomitas

Jelenleg `top_k * 2` jelolteket huzunk Chroma-bol (10 kandidat), majd RRF-rel top_k-t valasztunk. Kisebb dokumentumkorlnal (< 50 chunk) a `top_k * 1.5` (7-8 kandidat) is elegendo. Ezzel a Chroma-lekerdezes idejet ~25%-kal csokkentjuk. Alternativa: Chroma `hnsw:construction_ef` es `hnsw:search_ef` parameterek beallitasa a HNSW index epitesenel -- ez es > 500 chunk eseten erezheto.

### 3. (Bonusz) Async batch-szinten concurrent futas

A `graph.invoke()` szinkron. LangGraph tamogatja az `ainvoke()`-ot is; aszinkron koncurrens 10-20 kerdes indithato `asyncio.gather()`-rel. A ChromaDB SQLite alapu, konkurrens read rendben van, de a 100 query terheles > 5x gyorsabban kifuthet. Implementacios kockazat: a sentence-transformers modell CPU-thread-re szorul (GIL), igy a gyakorlati nyereseg ~2-3x.
