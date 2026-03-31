# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    TTS_MODELS_DIR=/app/models \
    HF_HOME=/app/models/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra all --no-install-project \
    && .venv/bin/python -m ensurepip

COPY scripts ./scripts
COPY tts_gateway ./tts_gateway

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra all \
    && uv pip install --python .venv/bin/python \
      'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'

ARG PRELOAD_KOKORO=false
ARG PRELOAD_POCKET=false

RUN PRELOAD_KOKORO=${PRELOAD_KOKORO} PRELOAD_POCKET=${PRELOAD_POCKET} \
    .venv/bin/python scripts/preload_models.py

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD [".venv/bin/python", "scripts/healthcheck.py"]

CMD ["sh", "-c", "uvicorn tts_gateway.main:create_app --factory --host ${TTS_GATEWAY_HOST:-0.0.0.0} --port ${TTS_GATEWAY_PORT:-8080}"]
