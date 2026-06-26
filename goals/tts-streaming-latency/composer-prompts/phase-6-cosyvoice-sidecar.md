# Composer Task: Phase 6 CosyVoice Sidecar Backend

You are implementing one slice in `/Users/andypai/Projects/tts-gateway`.
Codex is the orchestrator/reviewer. Leave a diff only; do not commit, push, or
open a PR.

## Context

The gateway now has:

- `StreamingTtsEngine` protocol in `tts_gateway/engines/base.py`
- `/tts/stream` and `/tts/stream/pcm`
- renderer support for native engine streaming via `stream_synthesize()`
- benchmark and transport validation scripts

This task should add CosyVoice as a first-class sidecar-backed engine. Do not
vendor or import the real CosyVoice model package. The gateway should talk to a
local sidecar over HTTP so the heavy runtime remains outside this process.

## Sidecar Contract For This Slice

Implement against this minimal local contract:

- `GET /health` returns JSON. Any 2xx means reachable.
- `POST /v1/tts/stream` accepts JSON:

```json
{"text":"...", "voice":"optional voice"}
```

- Successful stream response:
  - status `200`
  - `Content-Type: audio/raw`
  - `X-TTS-Sample-Rate`, `X-TTS-Channels`, and either
    `X-TTS-Sample-Width` or `X-TTS-Pcm-Format`
  - body is raw little-endian PCM bytes

Support `u8`, `s16le`, and `s32le` PCM formats. Buffer partial PCM frames so
each yielded `AudioChunk` contains complete frames.

## Implementation Requirements

- Add `cosyvoice` to the engine config surface.
- Add environment config:
  - `COSYVOICE_TTS_ENABLED` default `False`
  - `TTS_COSYVOICE_BASE_URL` default `http://127.0.0.1:50000`
  - `TTS_COSYVOICE_REQUEST_TIMEOUT_SECONDS` default should reuse the existing
    engine timeout if unset, or a clear positive integer default.
- Update CLI provider/fallback choices and env wiring for `tts serve` and
  `tts worker`.
- Add `tts_gateway/engines/cosyvoice_sidecar.py`.
- Implement `CosyVoiceSidecarEngine` as a `TtsEngine` that also satisfies the
  streaming protocol.
- `stream_synthesize()` should use `httpx.AsyncClient.stream()`.
- `synthesize()` should collect streamed chunks and merge them for buffered
  `/v1/speech` compatibility.
- Raise `EngineError` for unreachable sidecar, non-2xx responses, unsupported
  content type, unsupported PCM format, missing required audio headers, and
  partial trailing frames.
- Include health info for `cosyvoice` in `/health` without starting the real
  model.
- Keep Kokoro and Pocket behavior unchanged.
- Do not change route names or Raycast files.
- Do not change benchmark JSON files.

## Tests

Add focused automated tests that do not require a real model:

- config parses `cosyvoice` primary/fallback and rejects unknown engines
- CLI enables `COSYVOICE_TTS_ENABLED` when provider or fallback is `cosyvoice`
- runtime engine chain includes CosyVoice when configured
- `/health` includes CosyVoice state when disabled and enabled
- sidecar stream parses raw PCM headers and yields frame-aligned chunks
- sidecar stream buffers split frames across HTTP chunks
- missing/unsupported headers raise `EngineError`
- non-2xx sidecar response raises `EngineError`
- `synthesize()` merges streamed chunks for buffered endpoint compatibility
- renderer prefers CosyVoice native streaming when it is the primary engine

Use monkeypatching or fake `httpx.AsyncClient`/transport-style tests. Do not
start a real server unless the test is still fast and deterministic.

## Documentation

Add a short README or goal-note section describing the sidecar contract and the
local env vars. Keep it factual; do not claim real CosyVoice benchmarks until
Codex runs them later.

## Validation

Run these commands and report results:

```bash
uv run pytest tests/test_config.py tests/test_gateway_dual_mode.py tests/test_synthesis.py tests/test_api_integration.py tests/test_cosyvoice_sidecar.py --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Stop Rules

- Do not install model packages.
- Do not download model weights.
- Do not implement a CosyVoice sidecar server in this repo.
- Do not modify `/Users/andypai/Projects/raycast-tts-reader`.
- Do not commit, push, or open a PR.
- Do not read or print `.env` contents or secrets.
