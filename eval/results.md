# Funkcionalis ertekeles eredmenye

- LLM provider: **dummy**
- Osszesen: 15 kerdes
- Pass rate: **15/15 (100%)**
- Tool-sorrend egyezes: 14/15 (93%)
- Latency p50: 4 ms, p95: 23 ms, max: 23 ms

## Per-kerdes eredmenyek

| ID | Kat. | Pass | Tools | Latency (ms) |
|---|---|---|---|---|
| q01 | list | OK | list_documents [+] | 5 |
| q02 | list | OK | list_documents [+] | 2 |
| q03 | extract | OK | list_documents, get_extraction [+] | 4 |
| q04 | extract | OK | list_documents, get_extraction [+] | 3 |
| q05 | extract | OK | list_documents, get_extraction [+] | 3 |
| q06 | search | OK | list_documents, get_extraction [-] | 3 |
| q07 | search | OK | list_documents, search_documents [+] | 23 |
| q08 | search | OK | list_documents, search_documents [+] | 18 |
| q09 | compare | OK | list_documents, get_extraction, get_extraction, compare_documents [+] | 6 |
| q10 | compare | OK | list_documents, get_extraction, get_extraction, compare_documents [+] | 5 |
| q11 | compare | OK | list_documents, get_extraction, get_extraction, compare_documents [+] | 5 |
| q12 | validate | OK | list_documents, validate_document [+] | 4 |
| q13 | validate | OK | list_documents, validate_document [+] | 3 |
| q14 | validate | OK | list_documents, validate_document [+] | 3 |
| q15 | compare | OK | list_documents, get_extraction, get_extraction, compare_documents [+] | 5 |

## Kategoriankent

| Kategoria | Pass | Total |
|---|---|---|
| compare | 4 | 4 |
| extract | 3 | 3 |
| list | 2 | 2 |
| search | 3 | 3 |
| validate | 3 | 3 |
