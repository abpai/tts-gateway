# Composer Task: Phase 1 Latency Benchmarks

You are implementing the first slice of the approved goal in
`goals/tts-streaming-latency/goal.md`.

## Build

Finish the benchmark tooling for `tts-gateway`.

The current workspace may already contain a draft `scripts/bench_latency.py`
created by Codex. You may keep, edit, replace, or remove that draft as needed.
Treat the approved facts and plan as authoritative:

- `goals/tts-streaming-latency/facts.md`
- `goals/tts-streaming-latency/plan.md`

Implement:

1. A repeatable latency benchmark script at `scripts/bench_latency.py`.
2. Structured JSON output with:
   - run metadata
   - selected fixtures
   - raw per-run timings
   - per-fixture/per-endpoint summaries
   - optional comparison deltas against a prior report
3. Fixture coverage for short, sentence-length, medium, long, and markdown/noisy
   text.
4. Measurements for:
   - `/v1/speech` total time
   - `/tts/stream` time to first byte
   - `/tts/stream` total time
   - additional stream endpoints passed by CLI option, so future PCM/CosyVoice
     endpoints can be measured without rewriting the script
5. Warmup support through `POST /warmup`.
6. A cold/as-is/warm condition label. The script does not need to restart the
   server for a cold run; documenting the label behavior is enough.
7. Cache-bust support for synthesis-latency baselines, because `/v1/speech`
   reuses content-addressed artifacts.
8. Automated tests that do not require a live TTS server. Mock the HTTP
   transport or test pure helpers.
9. A concise benchmark usage note in either `README.md` or
   `goals/tts-streaming-latency/benchmarking.md`.

## Keep Scope Tight

Do not implement Raycast streaming playback in this task.
Do not implement gateway first-chunk tuning in this task.
Do not implement the streaming engine protocol in this task.
Do not implement CosyVoice in this task.
Do not commit, push, or open a PR.
Do not read, print, or modify `.env` files or secrets.

## Style

Follow `AGENTS.md`:

- Python 3.11+
- `uv`
- strict `ty`
- 2-space indent
- single quotes
- 88 character line width
- Pydantic `BaseModel` for interfaces/schemas
- small focused functions

## Validation

Run or leave ready to run:

```bash
uv run python scripts/bench_latency.py --help
uv run pytest tests/test_bench_latency.py --no-cov
uv run pytest
uv run ty check
```

If a command fails because of an environment issue, report the exact command and
failure in your final summary.
