# tts-gateway

A local text-to-speech gateway with a pluggable engine architecture. New open-source voice models ship constantly; tts-gateway gives clients a stable HTTP API with canonical `POST /v1/speech` and `POST /v1/jobs` endpoints, while retaining legacy `/tts` compatibility shims so swapping or adding models means implementing a small engine class, not rewiring your workflow.

Currently supports [Kokoro](https://github.com/hexgrad/kokoro) and [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) as native in-process engines, plus [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) via an optional HTTP sidecar. See [goals/tts-streaming-latency/cosyvoice-sidecar.md](goals/tts-streaming-latency/cosyvoice-sidecar.md) for the sidecar contract and env vars.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
# With Kokoro support (recommended)
uv tool install tts-gateway[kokoro]

# With Pocket TTS support
uv tool install tts-gateway[pocket]

# Both engines
uv tool install tts-gateway[all]
```

This installs a `tts` binary in `~/.local/bin/`.

### spaCy model (Kokoro only)

Kokoro depends on [misaki](https://github.com/hexgrad/misaki) for grapheme-to-phoneme conversion, which needs a spaCy English model. On first request, misaki tries to download `en_core_web_sm` via `spacy.cli.download`, but that shells out to `pip install` — which doesn't exist inside `uv tool` environments. You'll get a `SystemExit: 1` crash on the first TTS call.

Install the model manually into the tool's venv:

```bash
~/.local/share/uv/tools/tts-gateway/bin/python -m spacy download en_core_web_sm
```

### Upgrading

`uv tool upgrade` recreates the virtual environment, so the spaCy model must be reinstalled after every upgrade:

```bash
uv tool upgrade tts-gateway
~/.local/share/uv/tools/tts-gateway/bin/python -m spacy download en_core_web_sm
```

For local development, see [Development](#development) below.

## Docker

This repo now publishes a container image to GHCR from GitHub Actions.

```bash
docker pull ghcr.io/abpai/tts-gateway:latest
docker run --rm -p 8080:8080 \
  -e TTS_PRIMARY_ENGINE=kokoro \
  -e TTS_OUTPUT_FORMAT=mp3 \
  ghcr.io/abpai/tts-gateway:latest
```

The published image installs both native engine stacks and the Kokoro spaCy
model. By default it does not bake model weights into the image, so the first
`/warmup` or `/tts` request may still download engine weights unless you build a
preloaded image yourself.

To build a production image with baked model weights:

```bash
docker build \
  --build-arg PRELOAD_KOKORO=true \
  --build-arg PRELOAD_POCKET=false \
  -t tts-gateway:local .
```

Verify the container:

```bash
docker run --rm -d --name tts-gateway-test -p 8080:8080 tts-gateway:local
docker ps --filter name=tts-gateway-test
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/warmup
curl -X POST http://127.0.0.1:8080/v1/speech -F 'text=Hello world' -o output.mp3
```

For `bookmark.bunny`, the intended final-state deployment is to reference the
published image from Compose rather than vendoring this repo's Python source.

## Usage

Inspect or update the installed command:

```bash
tts --version
tts update
```

Start the server:

```bash
tts serve --provider kokoro
tts serve --provider kokoro --port 9000 --device cpu --format mp3
tts serve --provider kokoro --fallback pocket
```

Check the active gateway:

```bash
tts health
tts config
```

Speak text with streaming playback:

```bash
tts speak 'Hello world'
cat /tmp/speak_it.txt | tts speak
```

Streaming and playback are enabled by default. When stdout is redirected,
`tts speak` tees the audio to stdout while still playing it, so this saves a
file and plays it at the same time:

```bash
tts speak 'Hello world' > speech.mp3
```

Use `--output` to write a file without depending on shell redirection, or
`--no-play` to skip playback entirely:

```bash
tts speak 'Hello world' --no-play --output speech.mp3
cat /tmp/speak_it.txt | tts speak --no-play > speech.mp3
tts speak 'Hello world' --no-stream
```

`--no-play` without `--output` refuses to run when stdout is a terminal,
since it would otherwise dump binary audio to your screen; redirect stdout
or pass `--output` instead. Pressing Ctrl-C, or closing a pipe early (e.g.
piping into `head`), stops playback and exits cleanly.

When playing back with no `--output` and stdout is a terminal, `tts speak`
now requests raw PCM instead of MP3, so ffplay can start decoding on the
first chunk without waiting for MP3 frame boundaries — this cuts playback
start latency. Every other mode (writing to a file, teeing to a redirected
stdout, or `--no-play`) still requests MP3, since those sinks need MP3 bytes.

> **Version skew:** a 1.3.0 `tts speak` requires a 1.3.0 gateway — restart
> the server after updating, or the client's PCM/JSON requests won't match
> what an older server expects.

The client commands use `http://127.0.0.1:45123` by default. Set
`TTS_GATEWAY_URL` when the server uses a different address:

```bash
export TTS_GATEWAY_URL=http://127.0.0.1:45123
```

Synthesize speech:

```bash
# Canonical sync API
curl -X POST http://localhost:45123/v1/speech -F 'text=Hello world' -o output.mp3

# With a specific voice
curl -X POST http://localhost:45123/v1/speech -F 'text=Hello world' -F 'voice=af_heart' -o output.mp3

# With native speed control when the selected engine supports it
curl -X POST http://localhost:45123/v1/speech -F 'text=Hello world' -F 'speed=1.25' -o output.mp3

# Legacy compatibility route (deprecated shim, see below)
curl -X POST http://localhost:45123/tts -F 'text=Hello world' -o output.mp3

# Async job submission
curl -X POST http://localhost:45123/v1/jobs -F 'text=Hello world' | jq

# Chunk-level audio streaming (defaults to MP3)
curl -X POST http://localhost:45123/v1/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world","speed":1.25}' \
  -o output.mp3

# Raw PCM streaming (preferred for Raycast to avoid multi-chunk MP3 boundary risk)
curl -X POST http://localhost:45123/v1/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world","format":"pcm"}' \
  -o output.pcm
```

`/v1/speech/stream` reports the PCM layout on `X-TTS-Pcm-Format`,
`X-TTS-Sample-Rate`, and `X-TTS-Channels` response headers when
`format: "pcm"` is requested.

Raycast uses PCM-first streaming (`/v1/speech/stream` with `format: "pcm"`)
so playback can start on the first raw PCM chunk without stitching
independent MP3 frames at chunk joins.

`/tts/stream` and `/tts/stream/pcm` remain available as deprecated shims that
forward to `/v1/speech/stream`; they are slated for removal in v2 — migrate
to `/v1/speech/stream` directly.

Check server status:

```bash
curl http://localhost:45123/health
```

Pre-load models into memory:

```bash
curl -X POST http://localhost:45123/warmup
```

When both a primary and fallback engine are configured, the gateway tries the primary first and falls back on failure. Long texts are chunked automatically, synthesized concurrently across native chunks, and stitched into one final output file. The canonical API surface is `/v1/speech`, `/v1/speech/stream`, `/v1/jobs`, and `/v1/jobs/{key}/audio`; `/tts`, `/tts/sync`, `/tts/stream`, and `/tts/stream/pcm` remain available as deprecated compatibility shims slated for removal in v2.

## Running with PM2

For a persistent local server, use [PM2](https://pm2.keymetrics.io/):

```javascript
// ~/.pm2/ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "tts-gateway",
      script: "~/.local/bin/tts", // output of: which tts
      args: "serve --provider kokoro",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
      time: true,
    },
  ],
};
```

```bash
pm2 start ~/.pm2/ecosystem.config.js --only tts-gateway
pm2 logs tts-gateway
```

## Configuration

All settings can be controlled via environment variables. CLI flags take precedence (the CLI sets these env vars before starting the server).

| Variable                      | Default                       | Description                                    |
| ----------------------------- | ----------------------------- | ---------------------------------------------- |
| `TTS_PRIMARY_ENGINE`          | `kokoro`                      | Primary engine: `kokoro`, `pocket`, or `cosyvoice` |
| `TTS_FALLBACK_ENGINE`         | `none`                        | Fallback engine: `kokoro`, `pocket`, `cosyvoice`, or `none` |
| `TTS_OUTPUT_FORMAT`           | `mp3`                         | Output audio format: `wav` or `mp3`            |
| `TTS_DEVICE_MODE`             | `auto`                        | Torch device: `auto`, `cpu`, `mps`, `cuda`; Kokoro `auto` prefers CUDA then CPU, while `mps` remains opt-in |
| `TTS_DEFAULT_VOICE`           | _(none)_                      | Default voice name                             |
| `TTS_MODELS_DIR`              | `~/.cache/tts-gateway/models` | Model storage directory                        |
| `TTS_GATEWAY_HOST`            | `127.0.0.1`                   | Bind address                                   |
| `TTS_GATEWAY_PORT`            | `45123`                       | Bind port                                      |
| `TTS_CHUNK_MAX_CHARS`         | `500`                         | Max characters per chunk                       |
| `TTS_STREAM_FIRST_CHUNK_MAX_CHARS` | `180`                  | Max characters in the first stream chunk (time-to-first-audio) |
| `TTS_STREAM_CHUNK_MAX_CHARS`  | _(same as chunk max)_         | Max characters per subsequent stream chunk     |
| `TTS_REQUEST_TIMEOUT_SECONDS` | `3600`                        | Total request timeout                          |
| `TTS_ENGINE_TIMEOUT_SECONDS`  | `360`                         | Per-engine call timeout                        |
| `TTS_FFMPEG_PATH`             | `ffmpeg`                      | Path to ffmpeg binary (for MP3 encoding)       |
| `TTS_DATA_DIR`                | `~/.cache/tts-gateway/data`   | Job store and artifact directory               |
| `TTS_PIPELINE_VERSION`        | `1`                           | Cache-busting version for synthesis pipeline   |
| `TTS_WORKER_POLL_SECONDS`     | `1.0`                         | Background worker poll interval                |
| `TTS_WARMUP_ON_START`         | `true`                        | Start a non-blocking `/warmup` task during app startup |
| `KOKORO_TTS_ENABLED`          | `true`                        | Enable/disable Kokoro engine                   |
| `POCKET_TTS_ENABLED`          | `false`                       | Enable/disable Pocket TTS engine               |
| `COSYVOICE_TTS_ENABLED`       | `false`                       | Enable/disable CosyVoice sidecar engine        |
| `TTS_COSYVOICE_BASE_URL`      | `http://127.0.0.1:50000`      | CosyVoice sidecar base URL                     |
| `TTS_COSYVOICE_REQUEST_TIMEOUT_SECONDS` | _(engine timeout)_  | CosyVoice sidecar request timeout              |

Kokoro `TTS_DEVICE_MODE=auto` does not select MPS on Apple Silicon. For
sentence-length Kokoro-82M synthesis, measured CPU latency was 0.26-0.51s
versus 0.93-2.30s on MPS because per-shape kernel compilation dominates short
requests. Set `TTS_DEVICE_MODE=mps` explicitly to opt in.

## Development

```bash
make setup       # Create venv, install deps, set up pre-commit hooks
make test        # Run tests with coverage
make lint        # Run ruff linter with auto-fix
make format      # Run ruff formatter
make typecheck   # Run ty type checker
make run         # Start server (PROVIDER=kokoro by default)
```

`make setup` creates the local venv, installs dev dependencies plus all engine
extras, installs the Kokoro spaCy model, preloads engine weights, and sets up
pre-commit hooks. After it completes, the repo checkout is ready for real local
synthesis.

If you only want the dev toolchain without engine extras, use:

```bash
make install-dev
```

After that, you can verify the local server the same way as the container:

```bash
make run
curl http://127.0.0.1:45123/health
curl -X POST http://127.0.0.1:45123/warmup
curl -X POST http://127.0.0.1:45123/v1/speech -F 'text=Hello world' -o output.mp3
```

## Releasing

Use the repo helper to do the whole release flow in one command:

```bash
make release
```

That command:

1. bumps `project.version` in `pyproject.toml` by one patch version
2. runs lint, typecheck, tests, and packaging checks
3. commits the version bump
4. creates the matching git tag
5. pushes the branch and the tag

You can choose a different bump strategy:

```bash
make release BUMP=minor
make release BUMP=major
make release VERSION=0.2.0
```

To preview the exact commands first:

```bash
make release-dry-run
```
