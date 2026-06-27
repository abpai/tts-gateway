# Composer Task: Phase 9 Official CosyVoice FastAPI Bridge

You are implementing one narrow slice in `/Users/andypai/Projects/tts-gateway`.
Codex is the orchestrator/reviewer. Leave a diff only; do not commit, push, or
open a PR.

## Context

The gateway has a CosyVoice sidecar client that expects this contract:

- `GET /health`
- `POST /v1/tts/stream`
- JSON body: `{"text":"...", "voice":"optional"}`
- successful response: `Content-Type: audio/raw`, PCM headers, raw PCM body

The official CosyVoice FastAPI runtime has a different contract. The current
official `runtime/python/fastapi/server.py` exposes:

- `POST /inference_sft`
- form fields: `tts_text` and `spk_id`
- streaming response body: raw `int16` PCM bytes generated from model output
- default model server port: `50000`
- default model id in that script: `iic/CosyVoice2-0.5B`

Source inspected by Codex:
`https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/runtime/python/fastapi/server.py`

## Task

Add a lightweight bridge that adapts the official CosyVoice FastAPI runtime to
the `tts-gateway` sidecar contract. This should make the eventual real
CosyVoice benchmark path practical without changing the gateway client.

## Requirements

- Add `scripts/cosyvoice_official_bridge.py`.
- The bridge should be a FastAPI app runnable with:

```bash
uv run python scripts/cosyvoice_official_bridge.py \
  --upstream-base-url http://127.0.0.1:50001 \
  --host 127.0.0.1 \
  --port 50000 \
  --default-voice <voice-or-speaker-id>
```

- Expose `GET /health`.
  - Return JSON with bridge state, upstream base URL, upstream endpoint,
    sample rate, channels, PCM format, and default voice.
  - Optionally probe upstream docs/health if you keep it cheap and deterministic.
- Expose `POST /v1/tts/stream`.
  - Accept Pydantic request model with `text: str` and optional `voice: str`.
  - Reject empty/whitespace text with 422.
  - Forward to upstream `POST /inference_sft` as form data:
    `tts_text=<text>`, `spk_id=<voice or default voice>`.
  - Stream the upstream body to the client.
  - Set response headers:
    `Content-Type: audio/raw`
    `X-TTS-Sample-Rate`
    `X-TTS-Channels`
    `X-TTS-Pcm-Format: s16le`
    `X-TTS-Sample-Width: 2`
    `X-TTS-Upstream-Endpoint`
  - Handle non-2xx upstream responses as a clean gateway error response.
  - Handle upstream request failures as a clean gateway error response.
- Add CLI args for upstream base URL, upstream endpoint defaulting to
  `/inference_sft`, host, port, default voice, sample rate default `22050`,
  request timeout, and optional debug logging.
- Use existing dependencies (`fastapi`, `httpx`, `uvicorn`, `pydantic`).
- Do not import or install the real CosyVoice model package.
- Add tests in `tests/test_cosyvoice_official_bridge.py`.
  - Test health response.
  - Test successful stream forwards form data to upstream and returns audio/raw
    headers/body.
  - Test request voice overrides default voice.
  - Test empty text returns 422.
  - Test non-2xx upstream response becomes a non-2xx bridge response.
  - Test upstream request error becomes a non-2xx bridge response.
  - Test CLI parser defaults.
- Update `goals/tts-streaming-latency/cosyvoice-sidecar.md` with the bridge
  commands:
  - run official server on port `50001`
  - run bridge on port `50000`
  - run gateway with `tts serve --provider cosyvoice ...`
  - run benchmark with `--engine cosyvoice --require-engine-match`

## Non-goals

- Do not start/download/install CosyVoice.
- Do not change `tts_gateway/engines/cosyvoice_sidecar.py` unless absolutely
  needed for compatibility.
- Do not modify Raycast.
- Do not alter existing benchmark JSON.

## Validation

Run these commands and report results:

```bash
uv run python scripts/cosyvoice_official_bridge.py --help
uv run pytest tests/test_cosyvoice_official_bridge.py --no-cov
uv run pytest tests/test_cosyvoice_sidecar.py --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Stop Rules

- Do not commit, push, or open a PR.
- Do not read or print `.env` contents or secrets.
