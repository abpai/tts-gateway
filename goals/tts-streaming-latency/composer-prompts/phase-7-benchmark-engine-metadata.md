# Composer Task: Phase 7 Benchmark Engine Metadata

You are implementing one narrow slice in `/Users/andypai/Projects/tts-gateway`.
Codex is the orchestrator/reviewer. Leave a diff only; do not commit, push, or
open a PR.

## Context

The gateway now has:

- latency benchmark script: `scripts/bench_latency.py`
- Kokoro baseline JSON under `goals/tts-streaming-latency/benchmarks/`
- `/tts/stream` and `/tts/stream/pcm`
- CosyVoice sidecar config and engine

The execution plan includes a future live command:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --engine cosyvoice \
  --warmup \
  --output goals/tts-streaming-latency/benchmarks/cosyvoice.json
```

That command does not currently work because `bench_latency.py` has no
`--engine` option and does not record `/health` engine metadata.

## Task

Add optional engine metadata and health verification to `scripts/bench_latency.py`
so benchmark reports can explicitly label Kokoro vs CosyVoice runs and detect
when the requested engine does not match the live gateway.

## Requirements

- Add `--engine` with choices `kokoro`, `pocket`, `cosyvoice`.
- Add `engine` to the top-level JSON report as nullable/string.
- Fetch `GET /health` once per report and store a compact `health` object in
  the report.
- The health object should include at least:
  - whether the request succeeded
  - HTTP status or network error
  - `primaryEngine`
  - `fallbackEngine`
  - `engineChain`
  - `streamFirstChunkMaxChars`
  - `streamChunkMaxChars`
  - `engines` from the health payload if present
- If `--engine` is provided and `/health.primaryEngine` disagrees, do not abort
  by default, but record a warning in the report and print it in the summary.
- Add `--require-engine-match`; when this flag is set and the live primary
  engine disagrees with `--engine`, exit nonzero before running expensive
  benchmark requests.
- Preserve existing benchmark JSON compatibility: existing reports without
  `engine`, `health`, or `warnings` should still load with `load_report()`.
- Update tests in `tests/test_bench_latency.py`.
- Update `goals/tts-streaming-latency/benchmarking.md` examples to include the
  planned CosyVoice command and explain the engine-match flag.

## Non-goals

- Do not implement or start a real CosyVoice sidecar.
- Do not change existing benchmark JSON artifacts.
- Do not modify gateway runtime, routes, or Raycast files.
- Do not add network dependencies beyond stdlib.

## Validation

Run these commands and report results:

```bash
uv run python scripts/bench_latency.py --help
uv run pytest tests/test_bench_latency.py --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Stop Rules

- Do not commit, push, or open a PR.
- Do not read or print `.env` contents or secrets.
