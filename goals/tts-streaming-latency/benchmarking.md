# Latency Benchmarks

Use `scripts/bench_latency.py` to capture repeatable JSON latency reports for
`/v1/speech`, `/tts/stream`, and any additional stream endpoints you pass on the
CLI.

## Prerequisites

Start a live gateway, for example:

```bash
tts serve --provider kokoro --port 45123
```

## Capture a baseline

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --engine kokoro \
  --warmup \
  --cache-bust \
  --output goals/tts-streaming-latency/benchmarks/kokoro-baseline.json
```

## Capture a CosyVoice baseline

When the gateway is configured with CosyVoice as the primary engine:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --engine cosyvoice \
  --warmup \
  --cache-bust \
  --output goals/tts-streaming-latency/benchmarks/cosyvoice.json
```

Use `--require-engine-match` to abort before expensive benchmark requests when
the live gateway's `/health.primaryEngine` disagrees with `--engine`:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --engine cosyvoice \
  --require-engine-match \
  --warmup \
  --cache-bust \
  --output goals/tts-streaming-latency/benchmarks/cosyvoice.json
```

Without `--require-engine-match`, a mismatch is recorded in the report's
`warnings` array and printed in the summary instead of aborting.

## CosyVoice benchmark readiness

On June 26, 2026, no CosyVoice sidecar was reachable at the documented default
URL:

```bash
curl -fsS --max-time 2 http://127.0.0.1:50000/health
```

The benchmark guard was validated against the live Kokoro gateway on
`http://127.0.0.1:45123`:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --engine cosyvoice \
  --require-engine-match \
  --fixture short \
  --output /tmp/tts-engine-mismatch.json
```

It exited with code `1`, printed that the live primary engine was `kokoro`, and
did not create the output report. A real CosyVoice comparison still requires a
running sidecar plus a gateway configured with `TTS_PRIMARY_ENGINE=cosyvoice`.

## Real CosyVoice3 local sidecar spike

On June 26, 2026, a real Fun-CosyVoice3-0.5B zero-shot runtime was installed
outside this repo at `/tmp/tts-gateway-cosyvoice/CosyVoice` and run through the
gateway sidecar abstraction on an Apple M1 Max.

The official FastAPI server could load the model, but its zero-shot route failed
for CosyVoice3 because it converted `prompt_wav` to a tensor before calling
`inference_zero_shot`, while the CosyVoice3 frontend path tried to load that
tensor again as a WAV file. The local sidecar path in
`scripts/cosyvoice_local_sidecar.py` worked because it calls `AutoModel`
directly with a prompt WAV path and `stream=True`.

Runtime stack:

```bash
TOKENIZERS_PARALLELISM=false \
/tmp/tts-gateway-cosyvoice/.venv/bin/python \
  scripts/cosyvoice_local_sidecar.py \
  --cosyvoice-repo /tmp/tts-gateway-cosyvoice/CosyVoice \
  --model-dir /tmp/tts-gateway-cosyvoice/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --mode zero-shot \
  --host 127.0.0.1 \
  --port 50000 \
  --sample-rate 24000 \
  --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  --prompt-wav /tmp/tts-gateway-cosyvoice/CosyVoice/asset/zero_shot_prompt.wav

TTS_DATA_DIR=/tmp/tts-gateway-cosyvoice-real-data-local \
TTS_COSYVOICE_BASE_URL=http://127.0.0.1:50000 \
  uv run tts serve --provider cosyvoice --fallback none \
  --host 127.0.0.1 --port 45128 --format mp3
```

That spike intentionally used the bundled Chinese demo prompt WAV, so the
listening sample in
`goals/tts-streaming-latency/listening/cosyvoice3-zero-shot-m1-report.md` is
expected to sound Chinese-accented. Future English evaluation should use an
English narration reference WAV, its transcript, and `--english-narration` (see
`goals/tts-streaming-latency/cosyvoice-sidecar.md`).

Before benchmark capture, the sidecar was warmed with one direct synthesis:

```bash
curl -fsS -X POST http://127.0.0.1:50000/v1/tts/stream \
  -H 'Content-Type: application/json' \
  --data '{"text":"Short latency warmup."}' \
  --output /tmp/tts-gateway-cosyvoice/warmup.pcm
```

Reports:

- `goals/tts-streaming-latency/benchmarks/cosyvoice3-zero-shot-m1-short.json`
- `goals/tts-streaming-latency/benchmarks/cosyvoice3-zero-shot-m1-sentence.json`

Measured medians from one warmed sample:

| Fixture | Endpoint | First byte | Total | Delta vs Kokoro current |
| --- | --- | ---: | ---: | ---: |
| short | `/tts/stream` | 8363ms | 8398ms | +7594ms first byte |
| short | `/tts/stream/pcm` | 7127ms | 7128ms | +6506ms first byte |
| short | `/v1/speech` | 8794ms | 8794ms | +7202ms total |
| sentence | `/tts/stream` | 17953ms | 28011ms | +15663ms first byte |
| sentence | `/tts/stream/pcm` | 18424ms | 27124ms | +17540ms first byte |
| sentence | `/v1/speech` | 34216ms | 34217ms | +32383ms total |

This is enough evidence to keep Kokoro as the local Option+R default on this
machine. CosyVoice3 may still be worth exposing as an optional quality/voice
cloning backend, but not as the latency-optimized local default without a much
faster serving stack.

## Compare against a prior report

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --cache-bust \
  --compare goals/tts-streaming-latency/benchmarks/kokoro-baseline.json \
  --output goals/tts-streaming-latency/benchmarks/current.json
```

## Latest local transport check

On June 26, 2026, a warmed Kokoro/MPS gateway on
`http://127.0.0.1:45124` decoded both stream transports through `ffmpeg`:

```bash
uv run python scripts/check_stream_transport.py \
  --base-url http://127.0.0.1:45124 \
  --output-dir /tmp/tts-gateway-stream-transport
```

The same server produced
`goals/tts-streaming-latency/benchmarks/stream-transport-comparison.json`:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45124 \
  --warmup \
  --cache-bust \
  --stream-endpoint /tts/stream \
  --stream-endpoint /tts/stream/pcm \
  --compare goals/tts-streaming-latency/benchmarks/kokoro-baseline.json \
  --output goals/tts-streaming-latency/benchmarks/stream-transport-comparison.json
```

In that one-sample warmed run, PCM lowered first-byte latency versus chunked
MP3 for every fixture: short -108ms, medium -440ms, long -375ms, markdown
-593ms, and sentence -1331ms. Treat these as directional local measurements;
repeat runs and manual listening are still needed before making a default
recommendation.

After adding engine metadata to the benchmark script, a fresh current-code
Kokoro/MPS run on `http://127.0.0.1:45125` produced
`goals/tts-streaming-latency/benchmarks/kokoro-current-engine-metadata.json`:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45125 \
  --engine kokoro \
  --require-engine-match \
  --warmup \
  --cache-bust \
  --stream-endpoint /tts/stream \
  --stream-endpoint /tts/stream/pcm \
  --compare goals/tts-streaming-latency/benchmarks/kokoro-baseline.json \
  --output goals/tts-streaming-latency/benchmarks/kokoro-current-engine-metadata.json
```

The report health snapshot confirms `primaryEngine=kokoro`, `device=mps`,
`streamFirstChunkMaxChars=180`, and `streamChunkMaxChars=500`, with no warnings.
PCM again lowered first-byte latency versus chunked MP3 on every fixture:
short -147ms, medium -429ms, long -536ms, markdown -417ms, and sentence
-1406ms.

## Run conditions

The report includes a `condition` label:

| Label | Warmup behavior | Intended use |
| ----- | --------------- | ------------ |
| `as-is` | No `POST /warmup` | Default; measure the server in its current state |
| `warm` | Calls `POST /warmup` before measurements | Models loaded, steady-state latency |
| `cold` | No warmup | Label only; restart the gateway process before the run if you want a cold start |

`--warmup` is equivalent to `--condition warm`.

## Cache-sensitive routes

`/v1/speech` reuses content-addressed artifacts when the same text has already
been synthesized. Use `--cache-bust` for synthesis-latency baselines:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --warmup \
  --cache-bust
```

The report records `cacheBustToken` and hashes the final benchmark text so cache
hot runs are easy to distinguish from synthesis runs.

## Fixtures

Built-in fixture ids: `short`, `sentence`, `medium`, `long`, `markdown`.

Select a subset with repeated `--fixture` flags.

## Extra stream endpoints

The default stream set is `/tts/stream` and `/tts/stream/pcm`. Override it only
when narrowing or adding route variants:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --stream-endpoint /tts/stream/pcm
```

CosyVoice uses the same gateway stream endpoints when the live gateway is
configured with `TTS_PRIMARY_ENGINE=cosyvoice`; use `--engine cosyvoice` to
label and verify that run.

## JSON report shape

Each run writes:

- run metadata (`generatedAt`, `baseUrl`, `condition`, endpoints, `repeat`)
- optional `engine` label when `--engine` is supplied
- compact `health` snapshot from `GET /health` (engine chain, chunk limits, etc.)
- optional `warnings` (for example engine mismatch vs live gateway)
- selected fixtures with text hashes
- raw per-run timings in `runs`
- median summaries in `summary`
- optional `comparisons` when `--compare` is supplied

Measurements include `/v1/speech` total time, plus stream first-byte and total
time for every configured stream endpoint.

## Manual listening smoke workflow

Automated transport checks and latency JSON do not prove that streamed audio
sounds acceptable. Use `scripts/manual_stream_listening.py` to fetch live stream
payloads, validate ffmpeg decode, save replay commands, emit decoded WAV review
files with lightweight waveform sanity metrics, and write a Markdown report with
an unchecked human checklist.

Start a live gateway, then capture payloads and the report:

```bash
uv run python scripts/manual_stream_listening.py \
  --base-url http://127.0.0.1:45123 \
  --output-dir /tmp/tts-stream-listening
```

Optional flags:

- `--play` — run `ffplay` for each endpoint after decode validation (prefers
  the WAV review file when present)
- `--report /path/to/report.md` — override the default
  `<output-dir>/listening-report.md`
- `--endpoint /tts/stream/pcm` — repeat to limit endpoints

Each successful decode writes both the raw `.bin` payload and a normalized WAV
review file next to it (for example `wav_tts_stream.wav` and
`wav_tts_stream_pcm.wav`). The report includes `/health` metadata, primary
`ffplay` replay commands for the WAV files, raw-payload replay commands as a
fallback (with PCM `-f`, `-ar`, and `-ac` when needed), and a **Waveform
sanity** section with duration, peak, RMS, longest near-silence run, largest
adjacent-sample jump, and a conservative `PASS` / `WARN` / `SKIP` status.
Checklist items cover clicks, gaps, prompt start, Raycast stop/cancel, and
Option+R. Automated decode and waveform checks may show PASS, but human listening
stays **PENDING** until you edit the report and check the boxes after listening.

Record the verdict by opening the generated Markdown file, listening with the
replay commands (and Raycast for stop/cancel and Option+R), then checking the
boxes and updating the per-endpoint and overall **Status** lines from PENDING to
your conclusion.

A current-code Kokoro/MPS run on `http://127.0.0.1:45129` generated
`goals/tts-streaming-latency/listening/kokoro-current-report.md`. Automated
fetch, ffmpeg decode, WAV export, and waveform sanity passed for `/tts/stream`
and `/tts/stream/pcm`; the human listening verdict now passes in
`goals/tts-streaming-latency/listening/verdict.json`.

A real CosyVoice3 zero-shot run generated
`goals/tts-streaming-latency/listening/cosyvoice3-zero-shot-m1-report.md`.
That run used the bundled Chinese demo prompt WAV, so the sample is expected to
sound Chinese-accented. Automated fetch, ffmpeg decode, WAV export, and waveform
sanity passed for `/tts/stream` and `/tts/stream/pcm`; the human listening
verdict now passes in `goals/tts-streaming-latency/listening/verdict.json`.

For a single review surface with audio controls and a persistent checklist, open
`goals/tts-streaming-latency/listening/review.html` in a browser. It uses the
durable WAV files under `goals/tts-streaming-latency/listening/assets/`.

After listening, check the boxes in `review.html`, click **Copy Verdict JSON**,
save the payload (for example to `verdict.json`), and apply it to the goal
artifacts:

```bash
uv run python scripts/apply_listening_verdict.py --input verdict.json
```

The script writes `goals/tts-streaming-latency/listening/verdict.json`, updates
the Kokoro and CosyVoice listening reports, and sets `status` to `PASS` only when
every required checklist boolean is true. Goal completion still requires a human
to check the boxes; the script records that explicit verdict without inferring
pass from partial input.

Use `--dry-run` to preview the output JSON and report edits without writing
files.

## Goal artifact verification

Before human listening signoff, verify that required goal artifacts exist and
parse cleanly:

```bash
uv run python scripts/verify_goal_artifacts.py
```

After recording a complete human verdict in `listening/verdict.json`, require
it explicitly:

```bash
uv run python scripts/verify_goal_artifacts.py --require-human-verdict
```

## Validation

Use `--no-cov` for the targeted benchmark-script tests because the project
coverage threshold is intended for the full test suite:

```bash
uv run python scripts/bench_latency.py --help
uv run python scripts/manual_stream_listening.py --help
uv run python scripts/apply_listening_verdict.py --help
uv run python scripts/verify_goal_artifacts.py --help
uv run pytest tests/test_bench_latency.py --no-cov
uv run pytest tests/test_manual_stream_listening.py --no-cov
uv run pytest tests/test_apply_listening_verdict.py --no-cov
uv run pytest tests/test_verify_goal_artifacts.py --no-cov
uv run pytest tests/test_check_stream_transport.py --no-cov
uv run pytest
uv run ty check
```
