# CosyVoice Sidecar

The gateway can use CosyVoice through a local HTTP sidecar. The gateway process
does not load CosyVoice model weights; it forwards synthesis requests to a
separate service.

## Sidecar contract

- `GET /health` — returns JSON. Any 2xx response means the sidecar is reachable.
- `POST /v1/tts/stream` — JSON body:

```json
{"text": "...", "voice": "optional voice"}
```

Successful stream response:

- status `200`
- `Content-Type: audio/raw`
- `X-TTS-Sample-Rate`, `X-TTS-Channels`, and either `X-TTS-Sample-Width` or
  `X-TTS-Pcm-Format`
- body is raw little-endian PCM bytes

Supported PCM formats: `u8`, `s16le`, and `s32le`.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `COSYVOICE_TTS_ENABLED` | `false` | Enable the CosyVoice sidecar engine |
| `TTS_COSYVOICE_BASE_URL` | `http://127.0.0.1:50000` | Sidecar base URL |
| `TTS_COSYVOICE_REQUEST_TIMEOUT_SECONDS` | `TTS_ENGINE_TIMEOUT_SECONDS` | Per-request sidecar timeout |
| `TTS_PRIMARY_ENGINE` | `kokoro` | Set to `cosyvoice` to use the sidecar as primary |
| `TTS_FALLBACK_ENGINE` | `none` | Set to `cosyvoice` for sidecar fallback |

CLI example:

```bash
tts serve --provider cosyvoice --fallback kokoro
```

The CLI sets `COSYVOICE_TTS_ENABLED=true` when `cosyvoice` is the provider or
fallback.

## Gateway behavior

- `/tts/stream` and `/tts/stream/pcm` use native sidecar streaming when CosyVoice
  is the primary engine.
- `/v1/speech` and job synthesis collect streamed PCM and merge it for buffered
  output.
- `/health` reports CosyVoice as `disabled` or `sidecar` without loading a model.

This repo does not ship a CosyVoice model server. Use either:

- `scripts/cosyvoice_official_bridge.py` when the official CosyVoice FastAPI
  runtime works for your model and mode.
- `scripts/cosyvoice_local_sidecar.py` when you want to call CosyVoice
  `AutoModel` directly with `stream=True` (for example CosyVoice3 zero-shot,
  where the official FastAPI runtime currently mishandles prompt WAV paths).

### When to use which sidecar

| Use case | Sidecar |
| --- | --- |
| Official FastAPI runtime works for your model/mode | Official bridge |
| CosyVoice3 zero-shot (prompt WAV path bug in official runtime) | Local sidecar |
| You already run `runtime/python/fastapi/server.py` | Official bridge |
| Direct `AutoModel(...).inference_*` smoke test succeeded | Local sidecar |

Both sidecars expose the same gateway contract on port `50000` by default.

## Local CosyVoice sidecar

The local sidecar loads CosyVoice from a checkout you provide. Run it with the
CosyVoice Python environment, not the gateway venv. It calls model inference
with `stream=True` and converts each chunk to signed 16-bit little-endian PCM.

CosyVoice3 zero-shot on Apple Silicon (tested June 26, 2026):

```bash
# From the CosyVoice venv/interpreter:
python /path/to/tts-gateway/scripts/cosyvoice_local_sidecar.py \
  --cosyvoice-repo /path/to/CosyVoice \
  --model-dir /path/to/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --mode zero-shot \
  --host 127.0.0.1 \
  --port 50000 \
  --sample-rate 24000 \
  --english-narration \
  --prompt-text 'Transcript matching your English narration reference WAV.' \
  --prompt-wav /path/to/english-narration-reference.wav
```

The bundled CosyVoice `asset/zero_shot_prompt.wav` is a Chinese demo reference.
Use your own English narration WAV for English-accent evaluation; pass
`--english-narration` to fill default narration instruction text for instruct
modes. In zero-shot mode, `--prompt-text` must still be the transcript of your
English reference WAV.

CosyVoice2 and CosyVoice3 models use a 24kHz sample rate in the official
examples; pass `--sample-rate 24000` so gateway PCM headers match the model.

Supported `--mode` values:

| Mode | Required flags |
| --- | --- |
| `sft` (default) | `--default-voice` |
| `zero-shot` | `--prompt-text`, `--prompt-wav` |
| `cross-lingual` | `--prompt-wav` (English accent requires an English reference WAV) |
| `instruct` | `--default-voice`, `--instruct-text` (or `--english-narration`) |
| `instruct2` | `--instruct-text` (or `--english-narration`), `--prompt-wav` |

Start the gateway with CosyVoice as primary:

```bash
tts serve --provider cosyvoice --fallback kokoro --port 45123
```

Capture a real benchmark after the local sidecar and gateway are running:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --engine cosyvoice \
  --require-engine-match \
  --warmup \
  --cache-bust \
  --output goals/tts-streaming-latency/benchmarks/cosyvoice3-zero-shot-m1-short.json
```

The June 26, 2026 Apple M1 Max spike found that this local CosyVoice3 zero-shot
path works but is not latency-competitive with Kokoro for Option+R. The short
fixture reached first audio in 7.1s on `/tts/stream/pcm`; the sentence fixture
reached first audio in 18.4s on `/tts/stream/pcm`. See
`goals/tts-streaming-latency/benchmarking.md` for the full comparison.

## Official CosyVoice bridge

The bridge adapts the official CosyVoice FastAPI runtime to the sidecar
contract on port `50000` while the official server runs on port `50001`.

Supported `--mode` values map to official upstream endpoints:

| Mode | Upstream endpoint | Required bridge flags |
| --- | --- | --- |
| `sft` (default) | `/inference_sft` | `--default-voice` |
| `zero-shot` | `/inference_zero_shot` | `--prompt-text`, `--prompt-wav` |
| `cross-lingual` | `/inference_cross_lingual` | `--prompt-wav` |
| `instruct` | `/inference_instruct` | `--default-voice`, `--instruct-text` |
| `instruct2` | `/inference_instruct2` | `--instruct-text`, `--prompt-wav` |

Use `--upstream-endpoint` only when overriding the mode-derived path.

1. Start the official CosyVoice FastAPI server (outside this repo) on port
   `50001`:

```bash
# From a CosyVoice checkout with the model installed:
python runtime/python/fastapi/server.py --port 50001
```

2. Start the bridge on port `50000`.

SFT mode (speaker id from the loaded model):

```bash
uv run python scripts/cosyvoice_official_bridge.py \
  --mode sft \
  --upstream-base-url http://127.0.0.1:50001 \
  --host 127.0.0.1 \
  --port 50000 \
  --default-voice <speaker-id-from-model>
```

Zero-shot mode for CosyVoice2/CosyVoice3 (English narration reference WAV):

```bash
uv run python scripts/cosyvoice_official_bridge.py \
  --mode zero-shot \
  --upstream-base-url http://127.0.0.1:50001 \
  --host 127.0.0.1 \
  --port 50000 \
  --english-narration \
  --prompt-text 'Transcript of your English narration reference WAV.' \
  --prompt-wav /path/to/english-narration-reference.wav
```

The June 2026 CosyVoice3 spike used the bundled Chinese demo prompt WAV; that
listening sample is expected to sound Chinese-accented. Default English
evaluation should use an English narration reference WAV and its transcript
instead.

3. Start the gateway with CosyVoice as primary:

```bash
tts serve --provider cosyvoice --fallback kokoro --port 45123
```

4. Capture a real CosyVoice benchmark after the official runtime, bridge, and
   gateway are all running:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45123 \
  --engine cosyvoice \
  --require-engine-match \
  --warmup \
  --cache-bust \
  --output goals/tts-streaming-latency/benchmarks/cosyvoice.json
```

## Bridge contract smoke

On June 26, 2026, the bridge path was validated with a fake official upstream
that implemented `POST /inference_sft` and streamed raw `int16` PCM. The stack
was:

- fake official upstream on `http://127.0.0.1:50001`
- `scripts/cosyvoice_official_bridge.py` on `http://127.0.0.1:50000`
- `tts-gateway` with `--provider cosyvoice` on `http://127.0.0.1:45127`

Automated transport validation passed:

```bash
uv run python scripts/check_stream_transport.py \
  --base-url http://127.0.0.1:45127 \
  --output-dir /tmp/tts-cosyvoice-contract-transport \
  --text 'CosyVoice bridge contract smoke test.'
```

An engine-verified benchmark was written to
`goals/tts-streaming-latency/benchmarks/cosyvoice-bridge-contract-smoke.json`:

```bash
uv run python scripts/bench_latency.py \
  --base-url http://127.0.0.1:45127 \
  --engine cosyvoice \
  --require-engine-match \
  --warmup \
  --cache-bust \
  --fixture short \
  --output goals/tts-streaming-latency/benchmarks/cosyvoice-bridge-contract-smoke.json
```

This proves the gateway-to-bridge contract, not real CosyVoice quality or
latency. A real comparison still requires the official CosyVoice runtime and
model weights.
