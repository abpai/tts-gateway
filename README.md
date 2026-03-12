# tts-gateway

A local text-to-speech gateway with a pluggable engine architecture. New open-source voice models ship constantly; tts-gateway gives any client a stable `POST /tts` HTTP endpoint so swapping or adding models means implementing a small engine class, not rewiring your workflow.

Currently supports [Kokoro](https://github.com/hexgrad/kokoro) and [Pocket TTS](https://github.com/nicobailey/pocket-tts). Each engine runs natively in-process.

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
uv pip install \
  --python ~/.local/share/uv/tools/tts-gateway/bin/python \
  en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

For local development, see [Development](#development) below.

## Usage

Start the server:

```bash
tts serve --provider kokoro
tts serve --provider kokoro --port 9000 --device cpu --format mp3
tts serve --provider kokoro --fallback pocket
```

Synthesize speech:

```bash
# Basic
curl -X POST http://localhost:8000/tts -F 'text=Hello world' -o output.wav

# With a specific voice
curl -X POST http://localhost:8000/tts -F 'text=Hello world' -F 'voice=af_heart' -o output.wav

# MP3 output (if server started with --format mp3)
curl -X POST http://localhost:8000/tts -F 'text=Hello world' -o output.mp3
```

Check server status:

```bash
curl http://localhost:8000/health
```

Pre-load models into memory:

```bash
curl -X POST http://localhost:8000/warmup
```

When both a primary and fallback engine are configured, the gateway tries the primary first and falls back on failure. Long texts are chunked automatically and the resulting audio segments are stitched together.

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
| `TTS_PRIMARY_ENGINE`          | `kokoro`                      | Primary engine: `kokoro` or `pocket`           |
| `TTS_FALLBACK_ENGINE`         | `none`                        | Fallback engine: `kokoro`, `pocket`, or `none` |
| `TTS_OUTPUT_FORMAT`           | `wav`                         | Output audio format: `wav` or `mp3`            |
| `TTS_DEVICE_MODE`             | `auto`                        | Torch device: `auto`, `cpu`, `mps`, `cuda`     |
| `TTS_DEFAULT_VOICE`           | _(none)_                      | Default voice name                             |
| `TTS_MODELS_DIR`              | `~/.cache/tts-gateway/models` | Model storage directory                        |
| `TTS_GATEWAY_HOST`            | `127.0.0.1`                   | Bind address                                   |
| `TTS_GATEWAY_PORT`            | `8000`                        | Bind port                                      |
| `TTS_CHUNK_MAX_CHARS`         | `1400`                        | Max characters per chunk                       |
| `TTS_REQUEST_TIMEOUT_SECONDS` | `1200`                        | Total request timeout                          |
| `TTS_ENGINE_TIMEOUT_SECONDS`  | `360`                         | Per-engine call timeout                        |
| `TTS_FFMPEG_PATH`             | `ffmpeg`                      | Path to ffmpeg binary (for MP3 encoding)       |
| `KOKORO_TTS_ENABLED`          | `true`                        | Enable/disable Kokoro engine                   |
| `POCKET_TTS_ENABLED`          | `false`                       | Enable/disable Pocket TTS engine               |

## Development

```bash
make setup       # Create venv, install deps, set up pre-commit hooks
make test        # Run tests with coverage
make lint        # Run ruff linter with auto-fix
make format      # Run ruff formatter
make typecheck   # Run ty type checker
make run         # Start server (PROVIDER=kokoro by default)
```
