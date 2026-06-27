# Composer Task: Phase 3 Gateway Stream First-Chunk Planning

You are implementing the third slice of the approved goal in
`goals/tts-streaming-latency/goal.md`.

Work in this repo:

`/Users/andypai/Projects/tts-gateway`

## Why

The cache-busted Kokoro/MPS baseline at
`goals/tts-streaming-latency/benchmarks/kokoro-baseline.json` shows current
`/tts/stream` TTFA is still high, especially for long selections. The stream
route currently uses the same chunk plan as disk synthesis, so the first emitted
audio waits for a full normal-sized chunk. Add a stream-specific first-chunk
policy so long selections can produce first audio sooner.

## Hard constraints

1. Do not change `/v1/speech` or async job content hashes.
2. Do not add stream-only fields to `SynthesisSpec.to_json()`.
3. Do not change disk artifact cache layout or `synthesize_to_disk()` behavior.
4. Keep existing fallback engine ordering and ordered stream output.
5. Do not implement PCM transport, streaming engine protocols, or CosyVoice in
   this task.
6. Do not commit, push, or open a PR.

## Current files to inspect

- `tts_gateway/chunking.py`
- `tts_gateway/render.py`
- `tts_gateway/routes.py`
- `tts_gateway/config.py`
- `tts_gateway/types.py`
- `tests/test_chunking.py`
- `tests/test_synthesis.py`
- `tests/test_api_integration.py`
- `scripts/bench_latency.py`

## Build

Implement a stream-specific chunk plan:

1. Add config for stream chunking, preferably:
   - `TTS_STREAM_FIRST_CHUNK_MAX_CHARS`, default around `180`
   - `TTS_STREAM_CHUNK_MAX_CHARS`, default to `TTS_CHUNK_MAX_CHARS`
2. Expose those values in `/health`.
3. Add a pure helper that plans stream chunks using a smaller first chunk while
   preserving normalized text order.
4. Use the helper only in `/tts/stream` / `stream_audio`.
5. Keep the normal `plan_chunks(spec)` path unchanged for sync and job output.
6. Make the behavior deterministic for:
   - short text that already fits
   - several short sentences
   - one long first sentence
   - markdown/noisy text
   - no-punctuation prose
7. Update benchmark docs or plan notes only if the command surface changes.

## Validation

Run these in `/Users/andypai/Projects/tts-gateway`:

```bash
uv run pytest tests/test_chunking.py tests/test_synthesis.py tests/test_api_integration.py --no-cov
uv run pytest tests/test_bench_latency.py --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

If the live gateway at `http://127.0.0.1:45123` is available, optionally run a
cache-busted benchmark after implementation:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --warmup \
  --cache-bust \
  --compare goals/tts-streaming-latency/benchmarks/kokoro-baseline.json \
  --output /tmp/tts-gateway-stream-first-chunk.json
```

Leave the diff uncommitted for Codex to review.
