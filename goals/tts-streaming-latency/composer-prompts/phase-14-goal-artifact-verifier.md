# Composer Task: Phase 14 Goal Artifact Verifier

## Context

We are working in `/Users/andypai/Projects/tts-gateway` toward
`goals/tts-streaming-latency/goal.md`.

Most implementation is already present. The remaining final completion gate is
explicit human listening / Option+R signoff. Before that signoff, we need a
repo-native verifier that audits the non-human goal artifacts so Codex can
distinguish "ready for human verdict" from "missing evidence."

Existing relevant artifacts:

- `goals/tts-streaming-latency/goal.md`
- `goals/tts-streaming-latency/facts.md`
- `goals/tts-streaming-latency/plan.md`
- `goals/tts-streaming-latency/benchmarking.md`
- `goals/tts-streaming-latency/benchmarks/*.json`
- `goals/tts-streaming-latency/listening/*.md`
- `goals/tts-streaming-latency/listening/review.html`
- `goals/tts-streaming-latency/listening/assets/**`
- `scripts/apply_listening_verdict.py`
- `scripts/manual_stream_listening.py`

Follow repo style from AGENTS.md: Python 3.11+, `uv`, `ty`, Pydantic for
interfaces/schemas, 2-space indent, single quotes, focused functions.

## Task

Add a deterministic verifier for the goal artifacts.

## Requirements

1. Add `scripts/verify_goal_artifacts.py`.
2. The script should default to verifying `goals/tts-streaming-latency`.
3. It should read and validate:
   - `goal.md`, `facts.md`, and `plan.md` exist.
   - benchmark JSON files required for the current evidence exist:
     - `benchmarks/kokoro-baseline.json`
     - `benchmarks/stream-transport-comparison.json`
     - `benchmarks/kokoro-current-engine-metadata.json`
     - `benchmarks/cosyvoice3-zero-shot-m1-short.json`
     - `benchmarks/cosyvoice3-zero-shot-m1-sentence.json`
   - benchmark JSON files parse and include useful fields such as
     `generatedAt`, `fixtures`, `runs`, `summary`, and the relevant stream
     endpoints.
   - current Kokoro benchmark includes `/tts/stream/pcm` and health metadata
     showing `primaryEngine == "kokoro"`.
   - CosyVoice benchmark files are labeled `engine == "cosyvoice"` and compare
     against the current Kokoro baseline.
   - listening reports for Kokoro and CosyVoice exist, contain waveform sanity
     `status: PASS`, and still expose the human verdict section.
   - `listening/review.html` references audio files that exist on disk.
   - if `listening/verdict.json` exists, it parses and contains all required
     boolean checks used by `scripts/apply_listening_verdict.py`.
4. The script should print a compact human-readable audit with PASS/WARN/FAIL
   lines and return:
   - exit code `0` when required non-human artifacts pass and either no human
     verdict is present or a complete human verdict is present, as controlled
     below.
   - exit code `1` on missing/malformed required artifacts.
5. Add `--require-human-verdict`.
   - Without this flag, missing `listening/verdict.json` should be a warning,
     not a failure, so the script can verify "ready for human review."
   - With this flag, missing or incomplete human verdict should fail.
6. Add `--goal-dir PATH` for tests and future reuse.
7. Add focused tests in `tests/test_verify_goal_artifacts.py`.
   - Test a complete minimal artifact directory passes.
   - Test missing human verdict warns without `--require-human-verdict`.
   - Test missing human verdict fails with `--require-human-verdict`.
   - Test missing referenced review HTML audio file fails.
   - Test malformed benchmark JSON fails.
8. Update `goals/tts-streaming-latency/benchmarking.md` with a small
   "Goal artifact verification" section showing:
   - `uv run python scripts/verify_goal_artifacts.py`
   - `uv run python scripts/verify_goal_artifacts.py --require-human-verdict`

## Non-goals

- Do not mark the human verdict complete.
- Do not create `listening/verdict.json`.
- Do not edit benchmark JSON files.
- Do not run or start TTS/CosyVoice servers.
- Do not read, print, or commit secrets or `.env`.
- Do not commit or push.

## Validation

Run:

```bash
uv run python scripts/verify_goal_artifacts.py
uv run pytest tests/test_verify_goal_artifacts.py --no-cov
uv run ruff check scripts/verify_goal_artifacts.py tests/test_verify_goal_artifacts.py
uv run ty check
```

Leave the diff in the current workspace for Codex to review.
