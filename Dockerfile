# Lite image (default): rules, Presidio patterns, LSA embeddings, no PyTorch (~0.6 GB).
#   docker build -t rag-guardrails .
# Full image: bge embeddings, prompt-injection classifier, NLI cross-encoder and spaCy NER, all on CPU,
# with model weights baked in so cold starts never download anything.
#   docker build --build-arg WITH_ML=1 -t rag-guardrails:ml .
FROM python:3.12-slim

ARG WITH_ML=0
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 RAG_ROOT=/app HF_HOME=/app/models/hf PYTHONPATH=/app/src:/app
WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_ML" = "1" ]; then pip install -r requirements-ml.txt && python -m spacy download en_core_web_sm; fi

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY data/index ./data/index
COPY eval ./eval
COPY monitor ./monitor

# Warm up: fits/caches the LSA index and (WITH_ML=1) downloads every guard model into the image.
RUN python -c "from rag.service import RAGService; RAGService.build(trace_backend='none')"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn rag.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
