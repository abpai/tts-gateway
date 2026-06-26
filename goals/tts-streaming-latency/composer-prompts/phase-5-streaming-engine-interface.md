# Composer Task: Phase 5 Streaming-Capable Engine Interface

You are implementing one slice in `/Users/andypai/Projects/tts-gateway`.
Codex is the orchestrator/reviewer. Leave a diff only; do not commit, push, or
open a PR.

## Context

The gateway already has:

- `scripts/bench_latency.py`
- stream-specific chunk planning in `tts_gateway/chunking.py` and
  `tts_gateway/render.py`
- `/tts/stream` chunked MP3 streaming
- `/tts/stream/pcm` raw PCM streaming
- route and Raycast changes that prefer PCM streaming

The next goal is to make CosyVoice fit elegantly later by giving the renderer a
streaming-capable engine contract now. Do not implement CosyVoice in this task.

## Task

Introduce a capability-based engine contract so `tts_gateway.render` can prefer
native engine streaming when an engine supports it, while preserving the current
chunked synthesis fallback for Kokoro/Pocket engines.

## Requirements

- Add a `StreamingTtsEngine`-style protocol in `tts_gateway/engines/base.py`.
- Prefer `typing.Protocol` / duck typing over inheritance.
- Keep existing `TtsEngine` users working.
- Add a small helper/type guard if it keeps `render.py` simple.
- Update `stream_audio()` and `stream_pcm()` so streaming routes prefer a native
  streaming engine when the first available engine supports the streaming
  protocol.
- Preserve the existing chunked fallback path for engines that only implement
  `synthesize()`.
- Keep `/v1/speech` buffered artifact behavior unchanged.
- Keep the current first-chunk stream planner for non-streaming engines.
- For native streaming, pass the original request text and voice to the engine;
  do not pre-chunk text before calling the native stream.
- If a streaming engine fails before producing the first chunk, try the next
  available engine.
- If a streaming engine fails after bytes/chunks have already been yielded,
  propagate the error rather than switching voices mid-stream.
- Maintain cancellation/resource cleanup for stream generators.
- Keep methods/functions small and readable under the repo conventions.

## Tests

Add or update tests in `tests/test_synthesis.py` and/or focused helper tests:

- streaming-capable engine is preferred over a later synthesize-only engine
- synthesize-only engines still use the stream chunk planner
- streaming engine receives the full original text and requested voice
- streaming engine that fails before first chunk falls back to the next engine
- streaming engine that fails after first chunk propagates the error
- stream timeout behavior still raises instead of hanging
- `/v1/speech` still uses buffered synthesis and artifacts

Use existing test doubles in `tests/conftest.py` where possible; add small
focused doubles there only when needed.

## Validation

Run these commands and report results:

```bash
uv run pytest tests/test_synthesis.py tests/test_api_integration.py --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Stop Rules

- Do not implement CosyVoice.
- Do not modify `/Users/andypai/Projects/raycast-tts-reader`.
- Do not change benchmark JSON files.
- Do not change public route names.
- Do not read or print `.env` contents or secrets.
