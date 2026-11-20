FROM python:3.11-slim

LABEL org.opencontainers.image.title="multiomics-cancer-biomarker-ml" \
      org.opencontainers.image.description="Reproducible Snakemake + Python multi-omics cancer biomarker ML, SHAP explainability, and lifelines survival pipeline" \
      org.opencontainers.image.authors="Mallikarjuna Thippana" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && \
    apt-get install -y --no-install-recommends make ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY Snakefile Makefile README.md LICENSE CITATION.cff ./
COPY config ./config
COPY workflow ./workflow
COPY tests ./tests
COPY notebooks ./notebooks
RUN mkdir -p data/raw results figures

CMD ["snakemake", "--cores", "2", "--configfile", "config/ci.yaml", "--config", "_config_path=config/ci.yaml"]
