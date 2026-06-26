# Composer Review: Thermonuclear Gateway Review

You are reviewing the current uncommitted working tree in
`/Users/andypai/Projects/tts-gateway`.

This is a strict read-only adversarial review. Do not edit files, do not commit,
do not run destructive commands, and do not print secrets or `.env`.

## Scope

Review the full working tree, including untracked files. The changes implement
the TTS streaming latency goal:

- benchmark tooling and JSON goal artifacts
- stream first-chunk tuning
- PCM stream transport
- streaming-capable engine abstraction
- CosyVoice sidecar engine and sidecar scripts
- manual listening harness and verdict recorder
- goal artifact verifier
- docs and tests

Important: `git diff` does not include all new files. You must inspect
untracked files listed by `git status --short`, especially:

- `scripts/*.py`
- `tests/test_*.py`
- `tts_gateway/engines/cosyvoice_sidecar.py`
- `goals/tts-streaming-latency/**`

## Review Rubric

Lead with actionable findings only. Be harsh. Look for:

- correctness bugs and regressions
- streaming hangs, resource leaks, request cancellation problems
- HTTP/streaming contract mismatches
- broken fallback behavior or broken `/v1/speech`
- benchmark scripts that measure the wrong thing
- verifier/listening tools that can falsely pass
- CosyVoice sidecar edge cases and timeout mistakes
- tests that look green but do not cover the claimed behavior
- security / secret handling issues
- docs that claim behavior not implemented

Do not spend space on style-only comments unless they hide real maintenance
risk.

## Output Format

Return:

1. `Findings` grouped by severity, with file/line evidence.
2. `Required Fixes`.
3. `Review Verdict`: `clean`, `needs fixes`, or `blocked by missing context`.

If you find no release-blocking issues, say that directly and list any residual
risks.
