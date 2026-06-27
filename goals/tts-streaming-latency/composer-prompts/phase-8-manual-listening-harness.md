# Composer Task: Phase 8 Manual Listening Harness

You are implementing one narrow slice in `/Users/andypai/Projects/tts-gateway`.
Codex is the orchestrator/reviewer. Leave a diff only; do not commit, push, or
open a PR.

## Context

The active goal is not complete until streaming audio quality is manually
validated. The repo already has:

- `scripts/check_stream_transport.py` for fetching `/tts/stream` and
  `/tts/stream/pcm`, saving payloads, and validating decode with `ffmpeg`
- `scripts/bench_latency.py` for repeatable latency JSON
- Raycast streaming-first playback that prefers `/tts/stream/pcm`
- goal docs under `goals/tts-streaming-latency/`

The remaining gap is a repeatable manual listening workflow that leaves an
artifact instead of an informal memory. Codex cannot hear audio; the artifact
must make that limitation explicit and make it easy for a human to fill in the
verdict.

## Task

Add a script and tests for a manual listening smoke workflow.

## Requirements

- Add `scripts/manual_stream_listening.py`.
- The script should:
  - accept `--base-url` (default `http://127.0.0.1:45123`)
  - accept repeated `--endpoint`, defaulting to `/tts/stream` and
    `/tts/stream/pcm`
  - accept `--text`, with a default multi-sentence sample long enough for 3+
    chunks when stream chunking is active
  - accept `--output-dir` for saved payloads/report files
  - accept `--ffmpeg-path` and `--ffplay-path`
  - accept `--play` to play each fetched payload through `ffplay`
  - accept `--report` path, defaulting under `output-dir`
  - fetch each endpoint from the live gateway, save the payload, decode-check
    it with the same semantics as `check_stream_transport.py`, and emit a
    Markdown report
  - include `/health` metadata in the report when available
  - include exact replay commands for each payload, including PCM `-f`, `-ar`,
    and `-ac` arguments
  - include unchecked checklist lines for a human verdict, such as no clicks,
    no gaps, starts promptly, stop/cancel checked in Raycast, Option+R checked
  - never mark human listening as passed automatically
  - return nonzero when fetch/decode/play fails
- Prefer importing/reusing helpers from `scripts/check_stream_transport.py`
  instead of duplicating decoding command logic.
- Add `tests/test_manual_stream_listening.py`.
- Update `goals/tts-streaming-latency/benchmarking.md` or a focused goal doc
  with the manual listening command and how to record the verdict.

## Non-goals

- Do not modify gateway runtime/routes.
- Do not modify `/Users/andypai/Projects/raycast-tts-reader`.
- Do not pretend automated decode validation proves audio quality.
- Do not require interactive stdin in normal testable paths.

## Validation

Run these commands and report results:

```bash
uv run python scripts/manual_stream_listening.py --help
uv run pytest tests/test_manual_stream_listening.py --no-cov
uv run pytest tests/test_check_stream_transport.py --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Stop Rules

- Do not commit, push, or open a PR.
- Do not read or print `.env` contents or secrets.
