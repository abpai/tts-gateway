# Composer Task: Phase 11 Local CosyVoice Sidecar

## Context

The active goal is `goals/tts-streaming-latency/goal.md`.

We have:

- `tts_gateway/engines/cosyvoice_sidecar.py`, which expects the sidecar
  contract:
  - `GET /health`
  - `POST /v1/tts/stream` with JSON `{"text": "...", "voice": "optional"}`
  - streaming `audio/raw` response with sample-rate/channels/PCM headers
- `scripts/cosyvoice_official_bridge.py`, which adapts the official FastAPI
  runtime.

The official FastAPI runtime currently has a practical incompatibility for
CosyVoice3 zero-shot on this machine: it loads the uploaded prompt WAV into a
tensor, then the CosyVoice3 frontend path tries to load that tensor again as a
file. A direct smoke test proved the model works when calling:

```python
AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
model.inference_zero_shot(text, prompt_text, './asset/zero_shot_prompt.wav', stream=True)
```

We need a local sidecar script that keeps CosyVoice optional and out of
`tts-gateway` imports, but can be run from a CosyVoice Python environment for
real benchmarks.

## Task

Add `scripts/cosyvoice_local_sidecar.py`.

It should:

1. Preserve the gateway sidecar contract:
   - `GET /health`
   - `POST /v1/tts/stream`
   - JSON request: `{"text": "...", "voice": "optional"}`
   - response `audio/raw`
   - headers:
     - `X-TTS-Sample-Rate`
     - `X-TTS-Channels`
     - `X-TTS-Pcm-Format: s16le`
     - `X-TTS-Sample-Width: 2`
     - `X-TTS-Backend: cosyvoice-local`

2. Load CosyVoice lazily at app startup/runtime from an explicit repo path:
   - CLI `--cosyvoice-repo /path/to/CosyVoice`
   - add `<repo>/third_party/Matcha-TTS` and `<repo>` to `sys.path` before
     importing `cosyvoice.cli.cosyvoice.AutoModel`
   - CLI `--model-dir`, passed through to `AutoModel(model_dir=...)`
   - run this script with the CosyVoice venv/interpreter; do not add CosyVoice
     dependencies to `pyproject.toml`

3. Add CLI `--mode` with choices:
   - `sft`
   - `zero-shot`
   - `cross-lingual`
   - `instruct`
   - `instruct2`

4. Add mode-specific CLI flags and validation:
   - `--default-voice` for `sft` and `instruct`
   - `--prompt-text` for `zero-shot`
   - `--prompt-wav` for `zero-shot`, `cross-lingual`, and `instruct2`
   - `--instruct-text` for `instruct` and `instruct2`

5. Dispatch model calls with `stream=True`:
   - `sft`: `model.inference_sft(text, spk_id, stream=True)`
   - `zero-shot`: `model.inference_zero_shot(text, prompt_text, prompt_wav, stream=True)`
   - `cross-lingual`: `model.inference_cross_lingual(text, prompt_wav, stream=True)`
   - `instruct`: `model.inference_instruct(text, spk_id, instruct_text, stream=True)`
   - `instruct2`: `model.inference_instruct2(text, instruct_text, prompt_wav, stream=True)`

6. Convert each model output item to raw signed 16-bit little-endian PCM:
   - read `item['tts_speech']`
   - handle torch-like tensors via `detach().cpu().numpy()` if present
   - handle numpy arrays directly
   - clamp to `[-1, 1]`
   - multiply by `2**15 - 1`
   - cast to little-endian int16
   - yield bytes

7. `/health` should report:
   - `status: ok`
   - `backend: cosyvoice-local`
   - mode
   - modelDir
   - sampleRate
   - prompt/instruct/default voice configured booleans
   - prompt WAV basename only, not absolute path

8. Keep request validation strict:
   - reject blank request text with 422 via Pydantic
   - mode-required settings should fail at startup with clear `SystemExit`

9. Add tests in `tests/test_cosyvoice_local_sidecar.py` that do not import real
   CosyVoice:
   - settings validation
   - sys.path setup/import can be monkeypatched
   - health shape with no absolute prompt path
   - each mode dispatches to the expected fake model method with `stream=True`
   - PCM conversion clamps/casts correctly
   - `/v1/tts/stream` streams bytes and headers

10. Update `goals/tts-streaming-latency/cosyvoice-sidecar.md`:
    - document when to use the official bridge vs local sidecar
    - include the local sidecar command for the tested CosyVoice3 zero-shot
      setup
    - include the gateway + benchmark command sequence
    - mention that the local sidecar calls `stream=True`

## Non-goals

- Do not change `tts_gateway/engines/cosyvoice_sidecar.py` unless necessary for
  the existing sidecar contract.
- Do not add CosyVoice or torch dependencies to `pyproject.toml`.
- Do not start real servers, download models, or run real CosyVoice in tests.
- Do not commit changes.
- Do not read, print, or modify secrets or `.env` files.

## Style

Follow the repo AGENTS instructions:

- 2-space indent
- single quotes
- 88 char lines
- strict typing where practical for an optional external-runtime script
- small functions
- Pydantic for request schema
- comments only for why

## Validation

Run:

```bash
uv run python scripts/cosyvoice_local_sidecar.py --help
uv run pytest tests/test_cosyvoice_local_sidecar.py --no-cov
uv run ruff check scripts/cosyvoice_local_sidecar.py tests/test_cosyvoice_local_sidecar.py
uv run ty check
```

Leave a concise summary of changed files and validation results.
