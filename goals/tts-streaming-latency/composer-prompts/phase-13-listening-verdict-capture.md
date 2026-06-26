# Composer Task: Phase 13 Listening Verdict Capture

## Context

The active goal is `goals/tts-streaming-latency/goal.md`.

The final remaining completion gap is human listening validation. We now have:

- `goals/tts-streaming-latency/listening/review.html`
- `goals/tts-streaming-latency/listening/kokoro-current-report.md`
- `goals/tts-streaming-latency/listening/cosyvoice3-zero-shot-m1-report.md`
- durable WAV files under `goals/tts-streaming-latency/listening/assets/`

The HTML page has a checklist and a "Copy Verdict JSON" button. We need a
repo-native way to apply that explicit human verdict back into goal artifacts.

## Task

Add a small script that takes the copied verdict JSON and records it without
inventing or inferring a pass.

## Required behavior

1. Add `scripts/apply_listening_verdict.py`.

2. CLI behavior:
   - Read JSON from `--input path` or stdin.
   - Default output JSON:
     `goals/tts-streaming-latency/listening/verdict.json`
   - Default reports:
     - `goals/tts-streaming-latency/listening/kokoro-current-report.md`
     - `goals/tts-streaming-latency/listening/cosyvoice3-zero-shot-m1-report.md`
   - Support `--dry-run`.

3. Expected input shape should match `review.html` output:

```json
{
  "reviewedAt": "2026-06-26T19:00:00.000Z",
  "notes": "...",
  "clicks": true,
  "gaps": true,
  "prompt": true,
  "stop": true,
  "option-r": true,
  "kokoro-default": true
}
```

4. Validation:
   - Reject invalid JSON.
   - Reject missing required booleans.
   - Reject `reviewedAt` missing/blank.
   - Do not treat a partial checklist as pass.

5. Output verdict JSON should include:
   - original booleans and notes
   - `complete: true` only if every required boolean is true
   - `status: "PASS"` when complete, otherwise `"PARTIAL"`
   - `missingChecks` listing false checks
   - `appliedAt` in UTC ISO format

6. Report updates:
   - If complete, replace overall human verdict status lines in both reports
     with a clear PASS line.
   - If partial, replace overall status with PARTIAL and list missing checks.
   - Do not edit endpoint-specific PENDING sections unless the verdict is
     complete. On complete, endpoint-specific status lines may become PASS.
   - Append a short "Recorded human verdict" section to each report containing
     `reviewedAt`, `appliedAt`, `status`, and notes if nonblank.
   - Preserve the automated checks and waveform details.
   - Running the script twice should be idempotent: update/replace the recorded
     section rather than appending duplicates.

7. Add tests in `tests/test_apply_listening_verdict.py`.

8. Update `goals/tts-streaming-latency/benchmarking.md` near the listening
   section with:
   - open `review.html`
   - click "Copy Verdict JSON"
   - run `uv run python scripts/apply_listening_verdict.py --input verdict.json`
   - note that goal completion still requires a human to check the boxes.

## Non-goals

- Do not mark the current reports complete without an explicit input verdict.
- Do not modify Raycast or gateway behavior.
- Do not add dependencies.
- Do not commit changes.
- Do not read, print, or modify secrets or `.env` files.

## Validation

Run:

```bash
uv run python scripts/apply_listening_verdict.py --help
uv run pytest tests/test_apply_listening_verdict.py --no-cov
uv run ruff check scripts/apply_listening_verdict.py tests/test_apply_listening_verdict.py
uv run ty check
```

Leave a concise summary of changed files and validation results.
