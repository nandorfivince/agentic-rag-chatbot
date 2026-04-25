# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

# System fuggosegek: Tesseract OCR (magyar+angol+nemet), poppler (PDF render),
# curl a healthcheck-hez
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-hun tesseract-ocr-eng tesseract-ocr-deu \
        poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python csomagok telepitese -- elobb a torch CPU-only (kisebb image):
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install --no-cache-dir -r requirements.txt

# Sentence-transformers modell pre-download build-idoben.
# A futasi idoben nincs szukseg halozatra a modell-toltesre -- az elso query
# is gyors, nem csak a "warm-up" kovetkezik.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Alkalmazas forrasa
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit sajat aliaszolo: --server.address=0.0.0.0 hogy kulso halozatbol
# is elerheto legyen, --server.headless=true hogy ne probalkozzon browser-t
# nyitni a container-ben.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
