# Composer Task: Review Fixes for Gateway and CosyVoice Defaults

Workspace: `/Users/andypai/Projects/tts-gateway`

You are implementing fixes. Leave a diff only; do not commit, push, start real
TTS servers, download models, read secrets, or print `.env`.

## Context

Thermonuclear review found several gateway-side issues after the streaming
latency implementation:

1. `scripts/manual_stream_listening.py` exits success when waveform sanity is
   `WARN`; suspicious audio should fail the automation gate.
2. README does not document `/tts/stream/pcm` or stream chunk tuning env vars.
3. Stream routes prefetch the first chunk before returning `StreamingResponse`
   and do not check client disconnect around that prefetch.
4. CosyVoice examples/default operator path used the bundled Chinese
   zero-shot prompt. The user wants the CosyVoice default to be English
   accent/narration. Since zero-shot voice/accent comes from the prompt WAV,
   docs and helper defaults should steer users to English narration references,
   not the bundled Chinese demo.

## Requirements

### Manual listening strictness

- Update `workflow_failed()` so waveform `WARN` fails the command, not just
  `SKIP`.
- Add/update tests proving `WARN` and `SKIP` both fail while `PASS` succeeds.
- Keep reports human-readable; do not change existing completed verdict JSON.

### README streaming docs

- Document `/tts/stream/pcm` in README usage.
- Document `TTS_STREAM_FIRST_CHUNK_MAX_CHARS` and
  `TTS_STREAM_CHUNK_MAX_CHARS` in the env table.
- Mention that Raycast uses PCM-first streaming to avoid multi-chunk MP3
  boundary risk.

### Stream disconnect handling

- Add a practical disconnect guard around first-chunk prefetch in
  `tts_gateway/routes.py`.
- Keep route function complexity reasonable.
- If FastAPI `Request` injection is used, preserve existing API behavior and
  tests.
- Add route/unit tests for client-disconnected-before-stream where feasible
  without needing real engines. If this is too invasive, add a focused helper
  test around the guard.

### CosyVoice English narration defaults

- Make docs/examples default to an English narration reference, not the bundled
  Chinese demo prompt.
- Add a constant or helper where useful for a default English narration
  instruction text, e.g. "Read in a clear, neutral English narration voice."
- For `scripts/cosyvoice_local_sidecar.py` and
  `scripts/cosyvoice_official_bridge.py`, add an explicit
  `--english-narration` convenience flag or similarly small affordance if it
  reduces operator error:
  - It should supply English narration instruction text for instruct modes.
  - It must not invent zero-shot `--prompt-text`; that field is the transcript
    of the prompt WAV and must match the reference audio.
  - It must not pretend to supply an English accent for zero-shot without an
    English prompt WAV.
  - For zero-shot/cross-lingual/instruct2, require or clearly document an
    English reference WAV to get an English accent.
- Update tests for any new CLI behavior.
- Update `goals/tts-streaming-latency/benchmarking.md` and
  `goals/tts-streaming-latency/cosyvoice-sidecar.md` to explain that the
  previous CosyVoice3 spike used the bundled Chinese demo prompt and therefore
  the listening sample is expected to sound Chinese-accented; future/default
  English evaluation should use an English narration reference WAV and transcript.

## Non-goals

- Do not regenerate existing CosyVoice benchmark/listening audio artifacts.
- Do not mark old Chinese-prompt assets as English.
- Do not add real CosyVoice model dependencies.
- Do not change benchmark JSON.

## Validation

Run:

```bash
uv run pytest tests/test_manual_stream_listening.py tests/test_cosyvoice_local_sidecar.py tests/test_cosyvoice_official_bridge.py tests/test_api_integration.py --no-cov
uv run ruff check scripts/manual_stream_listening.py scripts/cosyvoice_local_sidecar.py scripts/cosyvoice_official_bridge.py tts_gateway/routes.py tests/test_manual_stream_listening.py tests/test_cosyvoice_local_sidecar.py tests/test_cosyvoice_official_bridge.py tests/test_api_integration.py
uv run ty check
```
