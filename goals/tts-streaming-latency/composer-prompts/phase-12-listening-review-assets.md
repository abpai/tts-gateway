# Composer Task: Phase 12 Listening Review Assets

## Context

The active goal is `goals/tts-streaming-latency/goal.md`.

We already have a manual listening harness:

- `scripts/manual_stream_listening.py`
- `tests/test_manual_stream_listening.py`

It fetches `/tts/stream` and `/tts/stream/pcm`, validates ffmpeg decode, saves
raw payload `.bin` files, and emits a Markdown report with replay commands and
PENDING human checklist items.

The remaining goal gap is manual streaming audio quality validation. We cannot
automatically mark that complete, but we can make the human review much easier
and improve objective evidence by producing decoded WAV review files plus a
small waveform sanity summary.

## Task

Enhance `scripts/manual_stream_listening.py` so every successfully decoded
stream payload also gets a normalized WAV review file and simple waveform
analysis.

## Required behavior

1. Add a decoded WAV file per endpoint:
   - For MP3 stream payloads, use ffmpeg to decode the payload to WAV.
   - For raw PCM stream payloads, use the response headers
     `X-TTS-Sample-Rate`, `X-TTS-Channels`, and `X-TTS-Pcm-Format` to decode
     to WAV.
   - Save next to the `.bin` payload in `--output-dir`.
   - Use deterministic names derived from the endpoint, for example:
     - `wav_tts_stream.wav`
     - `wav_tts_stream_pcm.wav`
   - Add the WAV path to the report.

2. Add replay commands for the WAV review files:
   - `ffplay -autoexit -nodisp -loglevel error <wav-path>`
   - The old raw-payload replay command should remain in the report as a lower
     level fallback.

3. Add a lightweight waveform sanity analysis for WAV files:
   - Use standard library `wave` plus `audioop` if useful, or `numpy` if that
     keeps the implementation clearer. The repo already depends on numpy.
   - Report:
     - duration seconds
     - sample rate
     - channels
     - sample width
     - peak absolute sample value
     - rms
     - longest near-silence run, using a conservative threshold such as
       absolute int16 <= 64
     - largest adjacent-sample jump
   - Add a status such as `PASS`, `WARN`, or `SKIP`:
     - `SKIP` when WAV export failed or no WAV exists
     - `WARN` for empty/near-empty audio, all-silent audio, extreme clipping,
       very long internal silence, or very large adjacent jump
     - `PASS` otherwise
   - Keep thresholds conservative and documented in code via names, not a long
     prose comment.

4. The report must still say automated checks do **not** validate perceived
   audio quality. Do not mark human listening as passed automatically.

5. `--play` behavior may continue to play raw payloads, but if a WAV review file
   exists, prefer playing the WAV review file because it is simpler and avoids
   header-dependent raw PCM playback.

6. Update dataclasses and tests as needed. Keep existing command-line options
   stable.

7. Update `goals/tts-streaming-latency/benchmarking.md` manual-listening section
   so it mentions WAV review assets and waveform sanity output.

## Non-goals

- Do not add a new external dependency.
- Do not remove raw `.bin` payload output.
- Do not change gateway behavior.
- Do not change Raycast code.
- Do not commit changes.
- Do not read, print, or modify secrets or `.env` files.

## Validation

Run:

```bash
uv run pytest tests/test_manual_stream_listening.py --no-cov
uv run ruff check scripts/manual_stream_listening.py tests/test_manual_stream_listening.py
uv run ty check
```

Leave a concise summary of changed files and validation results.
