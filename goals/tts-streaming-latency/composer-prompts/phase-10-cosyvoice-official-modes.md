# Composer Task: Phase 10 Official CosyVoice Bridge Modes

## Context

The active goal is `goals/tts-streaming-latency/goal.md`.

The gateway now has a CosyVoice sidecar client and a bridge script:

- `tts_gateway/engines/cosyvoice_sidecar.py`
- `scripts/cosyvoice_official_bridge.py`
- `tests/test_cosyvoice_official_bridge.py`
- `goals/tts-streaming-latency/cosyvoice-sidecar.md`

The current bridge only adapts the official `/inference_sft` endpoint:

```text
tts_text=<text>
spk_id=<voice>
```

That is contract-valid, but it is not enough for a real CosyVoice2/CosyVoice3
benchmark path. The official FastAPI runtime supports these endpoints:

- `/inference_sft`: `tts_text`, `spk_id`
- `/inference_zero_shot`: `tts_text`, `prompt_text`, file `prompt_wav`
- `/inference_cross_lingual`: `tts_text`, file `prompt_wav`
- `/inference_instruct`: `tts_text`, `spk_id`, `instruct_text`
- `/inference_instruct2`: `tts_text`, `instruct_text`, file `prompt_wav`

Official source used for this task:

`https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/runtime/python/fastapi/server.py`

## Task

Extend `scripts/cosyvoice_official_bridge.py` so it can adapt the official
CosyVoice FastAPI runtime in multiple modes while preserving the tts-gateway
sidecar contract:

```http
POST /v1/tts/stream
Content-Type: application/json

{"text": "...", "voice": "optional"}
```

The bridge should still respond with raw PCM plus these headers:

- `Content-Type: audio/raw`
- `X-TTS-Sample-Rate`
- `X-TTS-Channels`
- `X-TTS-Pcm-Format`
- `X-TTS-Sample-Width`
- `X-TTS-Upstream-Endpoint`

## Required behavior

1. Add a CLI `--mode` with choices:
   - `sft`
   - `zero-shot`
   - `cross-lingual`
   - `instruct`
   - `instruct2`

2. Derive the default official upstream endpoint from `--mode`:
   - `sft` -> `/inference_sft`
   - `zero-shot` -> `/inference_zero_shot`
   - `cross-lingual` -> `/inference_cross_lingual`
   - `instruct` -> `/inference_instruct`
   - `instruct2` -> `/inference_instruct2`

3. Keep `--upstream-endpoint` as an escape hatch, but make it optional. If
   supplied, normalize it like the current implementation does. If omitted, use
   the mode-derived endpoint.

4. Make `--default-voice` optional at argparse level, then validate it in
   `settings_from_args` only for modes that need `spk_id` (`sft` and
   `instruct`). It should still be used as the default speaker id when the
   gateway request does not include `voice`.

5. Add CLI options:
   - `--prompt-text` for `zero-shot`
   - `--prompt-wav` for `zero-shot`, `cross-lingual`, and `instruct2`
   - `--instruct-text` for `instruct` and `instruct2`

6. Validate mode-specific requirements in `settings_from_args`:
   - `sft`: requires `default_voice`
   - `zero-shot`: requires nonblank `prompt_text` and an existing readable
     `prompt_wav`
   - `cross-lingual`: requires an existing readable `prompt_wav`
   - `instruct`: requires `default_voice` and nonblank `instruct_text`
   - `instruct2`: requires nonblank `instruct_text` and an existing readable
     `prompt_wav`

7. Build the official upstream request according to mode:
   - `sft`: form data `tts_text`, `spk_id`
   - `zero-shot`: form data `tts_text`, `prompt_text`, multipart file
     `prompt_wav`
   - `cross-lingual`: form data `tts_text`, multipart file `prompt_wav`
   - `instruct`: form data `tts_text`, `spk_id`, `instruct_text`
   - `instruct2`: form data `tts_text`, `instruct_text`, multipart file
     `prompt_wav`

8. Include `mode`, `promptTextConfigured`, `promptWavConfigured`, and
   `instructTextConfigured` in `/health`. Do not expose local file contents.
   It is okay to expose only the basename of the prompt wav if useful, but do
   not include absolute paths in health.

9. Preserve the existing streaming cleanup behavior and non-2xx upstream error
   handling. Do not broaden exception catches beyond request errors.

10. Update `tests/test_cosyvoice_official_bridge.py` with focused tests for:
    - defaults are still SFT-compatible
    - mode-derived endpoint selection
    - mode-specific validation failures
    - zero-shot request sends prompt text and prompt WAV as multipart
    - instruct request sends `spk_id` and `instruct_text`
    - health includes the new mode/config booleans

11. Update `goals/tts-streaming-latency/cosyvoice-sidecar.md` with examples for:
    - SFT mode
    - zero-shot mode for CosyVoice2/CosyVoice3
    - the real benchmark command after starting the official runtime + bridge +
      gateway

## Non-goals

- Do not install, download, import, or run the real CosyVoice package.
- Do not change the gateway sidecar contract.
- Do not change `tts_gateway/engines/cosyvoice_sidecar.py` unless a test proves
  the bridge contract requires it.
- Do not commit changes.
- Do not read, print, or modify secrets or `.env` files.

## Style

Follow the repo AGENTS instructions:

- 2-space indent
- single quotes
- 88 char lines
- strict typing
- small functions
- Pydantic for request schemas
- custom/domain-specific errors where useful
- comments only for why

## Validation

Run:

```bash
uv run python scripts/cosyvoice_official_bridge.py --help
uv run pytest tests/test_cosyvoice_official_bridge.py --no-cov
uv run ruff check scripts/cosyvoice_official_bridge.py tests/test_cosyvoice_official_bridge.py
uv run ty check
```

Leave a concise summary of changed files and validation results.
