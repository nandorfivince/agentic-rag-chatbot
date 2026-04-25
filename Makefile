.PHONY: help install up down logs dev eval load test samples clean

help:
	@echo "Elerheto celok:"
	@echo "  make install   - Python fuggosegek telepitese (virtualenv-ben ajanlott)"
	@echo "  make samples   - Szintetikus minta dokumentumok generalasa (data/sample_docs/)"
	@echo "  make dev       - Streamlit lokalisan (docker nelkul, port 8501)"
	@echo "  make up        - Docker Compose indit (app + Ollama)"
	@echo "  make down      - Docker Compose megallit"
	@echo "  make logs      - Docker logok kovetese"
	@echo "  make eval      - Funkcionalis ertekeles (15 kerdes)"
	@echo "  make load      - Terheleses teszt (100 query)"
	@echo "  make test      - Unit tesztek (pytest)"
	@echo "  make clean     - Cache, pytest cache, chroma_db torlese"

install:
	pip install -r requirements.txt

samples:
	python data/generate_samples.py

dev:
	streamlit run app.py --server.port 8501

up:
	docker compose up -d
	@echo "UI elerheto: http://localhost:8501"
	@echo "Ollama elso modellhuzas: docker compose exec ollama ollama pull llama3.1:8b"

down:
	docker compose down

logs:
	docker compose logs -f

eval:
	python eval/run_eval.py

load:
	python load/benchmark.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	rm -f chroma_db/*.sqlite3
