# TTS Streaming Latency Plan

## Solution Approach

Make latency measurable first, then move the hot path from buffered file
generation to streaming playback. Because backward compatibility is not a goal,
prefer a simpler streaming-first design over preserving every existing endpoint
or Raycast preference behavior.

The gateway should evolve from a single `synthesize() -> AudioChunk` engine
contract to a capability-based engine contract where some engines can stream
incremental audio. Kokoro remains the tuned local baseline. CosyVoice becomes a
first-class sidecar-backed streaming engine, with its latency and quality
benchmarked against tuned Kokoro before any default-engine recommendation
changes. CosyVoice is worth this shape because the official project advertises
bi-streaming and latency as low as 150ms, but its runtime surface is heavy
enough that tts-gateway should keep it behind a stable sidecar boundary.

Relevant source notes:

- Current `tts-gateway` engines only expose `synthesize(text, voice) ->
  AudioChunk` in `tts_gateway/engines/base.py`.
- Current `/tts/stream` is chunk-complete streaming in
  `tts_gateway/render.py`, and the route waits for the first encoded chunk in
  `tts_gateway/routes.py`.
- Current Raycast playback buffers the full response in
  `/Users/andypai/Projects/raycast-tts-reader/src/tts-utils.ts`, writes a file
  in `src/play.ts`, and plays that file in `src/playback-controller.ts`.
- CosyVoice references: `https://github.com/FunAudioLLM/CosyVoice`,
  `https://funaudiollm.github.io/cosyvoice2/`, and
  `https://arxiv.org/abs/2505.17589`.

## Ordered Steps

### 1. Add latency benchmarks before behavior changes

Files/systems:

- `scripts/bench_latency.py`
- `goals/tts-streaming-latency/benchmarks/`
- `README.md` or a focused benchmark note under `goals/tts-streaming-latency/`

Implementation:

- Add fixtures for short, sentence-length, medium, long, and markdown/noisy
  text.
- Measure warm and cold runs.
- Measure `/v1/speech` total time.
- Measure `/tts/stream` time to first byte and total time.
- Leave extension points for PCM streaming and CosyVoice endpoints.
- Emit JSON so benchmark runs can be compared over time.
- Capture the current Kokoro/MPS baseline from `http://127.0.0.1:45123`.

Verification:

- `uv run python scripts/bench_latency.py --base-url http://127.0.0.1:45123 --warmup --cache-bust --output goals/tts-streaming-latency/benchmarks/kokoro-baseline.json`
- `uv run python scripts/bench_latency.py --base-url http://127.0.0.1:45123 --compare goals/tts-streaming-latency/benchmarks/kokoro-baseline.json`
- `uv run pytest tests/test_bench_latency.py --no-cov`
- `uv run pytest`
- `uv run ty check`

### 2. Make Raycast playback streaming-first

Files/systems:

- `/Users/andypai/Projects/raycast-tts-reader/src/tts-utils.ts`
- `/Users/andypai/Projects/raycast-tts-reader/src/play.ts`
- `/Users/andypai/Projects/raycast-tts-reader/src/playback-controller.ts`
- `/Users/andypai/Projects/raycast-tts-reader/src/read-selected-text.tsx`
- `/Users/andypai/Projects/raycast-tts-reader/src/read-text-editor.tsx`
- `/Users/andypai/Projects/raycast-tts-reader/package.json`

Implementation:

- Replace the gateway hot path with a streaming path that posts JSON to
  `/tts/stream`.
- Pipe `response.body` into `ffplay -nodisp -autoexit -loglevel error -i -`
  so playback starts before the full response is available.
- Because compatibility is not required, simplify or remove preference behavior
  that forces buffering if it makes the hot path harder to reason about.
- Link Stop Audio to both player termination and request abort.
- Add a small test harness for route selection and stream-to-player wiring. If
  the Raycast repo has no test runner, add a minimal Vitest setup around pure
  functions and mocked `fetch`/`spawn`.

Verification:

- `pnpm lint`
- `pnpm build`
- `pnpm test` after adding the test script
- Manual Option+R smoke with the live gateway.
- Confirm Stop Audio ends playback and cancels the stream request.
- Compare the Raycast-path first-audio latency against the baseline benchmark.

### 3. Add stream-specific first-chunk planning in tts-gateway

Files/systems:

- `tts_gateway/chunking.py`
- `tts_gateway/render.py`
- `tts_gateway/config.py`
- `tts_gateway/types.py`
- `tests/test_chunking.py`
- `tests/test_synthesis.py`
- `tests/test_api_integration.py`

Implementation:

- Do not rely only on global `TTS_CHUNK_MAX_CHARS`.
- Add a stream-specific chunk planner or first-chunk policy so long selections
  can produce first audio faster without changing buffered job chunk hashes by
  accident.
- Consider settings such as `TTS_STREAM_FIRST_CHUNK_MAX_CHARS` and
  `TTS_STREAM_CHUNK_MAX_CHARS`, or a simpler fixed first-sentence policy if
  the code stays clearer.
- Preserve deterministic ordering and existing fallback behavior within the
  gateway.

Verification:

- Unit tests for short text, many short sentences, one long first sentence,
  markdown/noisy text, and no-punctuation prose.
- Route tests proving `/tts/stream` uses the stream planner.
- Benchmark long-selection server TTFA before and after.
- `uv run pytest tests/test_chunking.py tests/test_synthesis.py tests/test_api_integration.py`
- `uv run ty check`

### 4. Validate and, if needed, replace the streaming transport

Files/systems:

- `tts_gateway/audio.py`
- `tts_gateway/render.py`
- `tts_gateway/routes.py`
- Raycast `playback-controller.ts`

Implementation:

- Test whether independently encoded MP3 chunks play cleanly through
  `ffplay -i -`.
- If multi-chunk MP3 has gaps, clicks, or decoder stalls, switch the streaming
  endpoint and Raycast player to raw PCM or a single long-lived ffmpeg encoder.
- Prefer PCM for lower latency and simpler framing if the Raycast path can own
  the playback command.

Verification:

- Manual listening test on 3+ chunk samples.
- Benchmark MP3 stream vs PCM stream first-byte and total time.
- Automated route test for the selected stream content type.
- `uv run pytest tests/test_synthesis.py tests/test_api_integration.py`
- `pnpm build`

### 5. Introduce streaming-capable engine interfaces

Files/systems:

- `tts_gateway/engines/base.py`
- `tts_gateway/render.py`
- `tts_gateway/runtime.py`
- `tts_gateway/engines/kokoro_native.py`
- `tts_gateway/engines/pocket_native.py`
- `tests/conftest.py`
- `tests/test_synthesis.py`
- `tests/test_gateway_dual_mode.py`

Implementation:

- Define a protocol for engines that can yield incremental audio frames or
  `AudioChunk` values.
- Keep a simple adapter path for engines that only support full synthesis.
- Update stream rendering to prefer native engine streaming when available and
  fall back to buffered chunk synthesis otherwise.
- Optionally expose Kokoro pipeline segment streaming as an intermediate local
  improvement.

Verification:

- Mock streaming engine tests for ordered output, fallback to non-streaming
  engines, timeout behavior, and no-engine failures.
- Regression tests proving `/v1/speech` still produces complete artifacts.
- `uv run pytest`
- `uv run ty check`

### 6. Implement CosyVoice as a sidecar-backed streaming backend

Files/systems:

- `tts_gateway/config.py`
- `tts_gateway/runtime.py`
- `tts_gateway/engines/cosyvoice_sidecar.py`
- `tts_gateway/engines/base.py`
- `pyproject.toml`
- `README.md`
- `Dockerfile` or PM2/runbook docs if needed
- `scripts/bench_latency.py`
- `tests/test_cosyvoice_sidecar.py`

Implementation:

- Add `cosyvoice` as a first-class engine option.
- Add config for sidecar base URL, request timeout, stream timeout, default
  voice/prompt mapping, and enabled state.
- Implement a sidecar client using `httpx` streaming so tts-gateway owns
  routing, health reporting, metrics, fallback chain behavior, and client API.
- Define the internal sidecar contract clearly. The first version can target a
  local CosyVoice service that returns PCM or WAV-compatible chunks.
- Add a mock sidecar server test so automated tests do not require the model.
- Document how to run the real CosyVoice sidecar and what hardware/runtime is
  expected.

Verification:

- Unit tests with a fake streaming sidecar.
- Health output includes CosyVoice state without requiring model startup in
  normal tests.
- Benchmark script includes a CosyVoice mode and writes comparable JSON.
- Live spike with a real sidecar when available:
  `uv run python scripts/bench_latency.py --base-url http://127.0.0.1:45123 --engine cosyvoice --warmup --output goals/tts-streaming-latency/benchmarks/cosyvoice.json`
- `uv run pytest`
- `uv run ty check`

### 7. Compare tuned Kokoro and CosyVoice, then decide defaults

Files/systems:

- `goals/tts-streaming-latency/benchmarks/*.json`
- `README.md`
- release notes / changelog if present
- Raycast onboarding text if behavior changes

Implementation:

- Compare warm Kokoro streaming, tuned stream-first Kokoro, optional Kokoro
  native streaming, and CosyVoice sidecar on the same fixtures.
- Keep Kokoro as default unless CosyVoice clearly improves the target
  Option+R workflow enough to justify its sidecar cost.
- Document breaking changes as a major release.

Verification:

- Benchmark comparison JSON checked into the goal artifact directory.
- Manual playback quality notes for Kokoro and CosyVoice.
- Final `make test` or equivalent gate:
  `uv run ruff check .`
  `uv run ruff format --check .`
  `uv run ty check`
  `uv run pytest`
  `pnpm --dir /Users/andypai/Projects/raycast-tts-reader lint`
  `pnpm --dir /Users/andypai/Projects/raycast-tts-reader build`
  `pnpm --dir /Users/andypai/Projects/raycast-tts-reader test`

## Risks And Open Questions

- CosyVoice may require a GPU/runtime setup that is not worth it for the local
  Option+R workflow, even if its model-level TTFA is excellent.
- CosyVoice streaming details may differ by version and serving backend. The
  sidecar contract should be tested with a fake server first, then a real model.
- Per-chunk MP3 may introduce audible seams. PCM streaming may be the cleaner
  breaking-change path.
- Raycast extension tests are not currently present, so adding a lightweight
  test runner is part of the work.
- Benchmarks should guide defaults, not unit-test thresholds. Normal tests
  should verify behavior; benchmark scripts should report latency.
- Because backward compatibility is optional, the implementation should avoid
  carrying old preference and endpoint complexity unless it directly serves the
  measured workflow.
