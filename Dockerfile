FROM python:3.11-slim

LABEL maintainer="Hospital AI Team"
LABEL description="Sistema Inteligente de Soporte Hospitalario"
LABEL version="2.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    API_HOST=0.0.0.0 \
    API_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY train_pipeline.py .
COPY generate_sample_data.py .
COPY streamlit_app.py .

RUN mkdir -p /app/data/raw /app/data/processed /app/data/incoming /app/models

VOLUME ["/app/data", "/app/models"]

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
