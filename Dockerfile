FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.hfcache \
    COPILOT_SOURCE=processed \
    COPILOT_EMBEDDER=pubmedbert \
    LLM_MODE=template

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('NeuML/pubmedbert-base-embeddings')"

COPY src/ ./src/
COPY artifacts/lightgbm_ranker.txt ./artifacts/lightgbm_ranker.txt
COPY data/processed/ ./data/processed/
COPY data/clinical/ ./data/clinical/
COPY deploy/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8080
CMD ["./entrypoint.sh"]
