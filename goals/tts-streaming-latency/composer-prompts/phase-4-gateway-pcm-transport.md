# Composer Task: Phase 4 Gateway PCM Transport

You are implementing the gateway half of Step 4 in
`goals/tts-streaming-latency/goal.md`.

Work in this repo:

`/Users/andypai/Projects/tts-gateway`

## Why

`/tts/stream` currently emits independently encoded MP3 chunks. That gives
first-byte latency improvements after Phase 3, but concatenated MP3 chunks may
have decoder stalls, gaps, or audible artifacts. Add an explicit raw PCM stream
endpoint so the Raycast client can choose a simpler low-latency transport.

## Current context

The worktree already contains uncommitted Phase 1 and Phase 3 changes:

- `scripts/bench_latency.py`
- `tts_gateway/chunking.py`
- `tts_gateway/render.py`
- `tts_gateway/config.py`
- `tts_gateway/routes.py`
- related tests

Preserve that work. Do not revert or rewrite unrelated goal files.

## Build

Implement gateway PCM streaming:

1. Add a `/tts/stream/pcm` route.
2. Reuse the stream-specific chunk plan from Phase 3.
3. Stream raw PCM bytes from ordered `AudioChunk` values.
4. Expose enough metadata for clients to play raw PCM:
   - `Content-Type: audio/raw`
   - `X-TTS-Mode: stream-pcm`
   - `X-TTS-Primary-Engine`
   - `X-TTS-Sample-Rate`
   - `X-TTS-Channels`
   - `X-TTS-Sample-Width`
   - `X-TTS-Pcm-Format` such as `s16le`, `u8`, or `s32le`
5. Keep `/tts/stream` as MP3 and preserve its existing behavior.
6. Keep `/v1/speech` and async jobs unchanged.
7. Add tests proving:
   - `/tts/stream/pcm` returns the raw PCM content type and metadata headers.
   - The PCM route uses the stream chunk planner.
   - MP3 `/tts/stream` still returns `audio/mpeg`.
   - Empty text and no-engine errors behave like the MP3 stream route.
8. Add a small transport validation script if it stays focused and useful,
   preferably `scripts/check_stream_transport.py`, that can fetch MP3/PCM stream
   endpoints from a live gateway, save the payload, and run `ffmpeg` decode-to-
   null checks. Automated tests for the script should not require a live server
   or model.
9. Update `goals/tts-streaming-latency/benchmarking.md` only if the command
   examples need to mention `/tts/stream/pcm`.

## Non-goals

- Do not update Raycast in this task.
- Do not implement CosyVoice.
- Do not introduce streaming engine protocols yet.
- Do not remove MP3 streaming.
- Do not commit, push, or open a PR.

## Validation

Run these in `/Users/andypai/Projects/tts-gateway`:

```bash
uv run pytest tests/test_synthesis.py tests/test_api_integration.py --no-cov
uv run pytest tests/test_bench_latency.py --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

If you add a transport script, also run its targeted tests/help command.

Leave the diff uncommitted for Codex review.
