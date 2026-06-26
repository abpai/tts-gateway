#!/usr/bin/env python3
"""Run CosyVoice directly as a tts-gateway sidecar without the official FastAPI runtime."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS_DIR))

from cosyvoice_defaults import (  # noqa: E402
  DEFAULT_ENGLISH_NARRATION_INSTRUCT,
)

_BACKEND_NAME = 'cosyvoice-local'
_DEFAULT_HOST = '127.0.0.1'
_DEFAULT_PORT = 50000
_DEFAULT_SAMPLE_RATE = 22050
_DEFAULT_CHANNELS = 1
_DEFAULT_PCM_FORMAT = 's16le'
_DEFAULT_SAMPLE_WIDTH = 2

SidecarMode = Literal['sft', 'zero-shot', 'cross-lingual', 'instruct', 'instruct2']


@dataclass(frozen=True)
class LocalSidecarSettings:
  mode: SidecarMode
  cosyvoice_repo: Path
  model_dir: Path
  default_voice: str | None
  prompt_text: str | None
  prompt_wav_path: Path | None
  instruct_text: str | None
  sample_rate: int
  channels: int
  pcm_format: str
  sample_width: int


class TtsStreamRequest(BaseModel):
  text: str
  voice: str | None = None

  @field_validator('text')
  @classmethod
  def text_not_blank(cls, value: str) -> str:
    if not value.strip():
      raise ValueError('text must not be empty or whitespace')
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--cosyvoice-repo',
    required=True,
    help='Path to the CosyVoice checkout used for imports',
  )
  parser.add_argument(
    '--model-dir',
    required=True,
    help='Model directory passed to AutoModel(model_dir=...)',
  )
  parser.add_argument(
    '--mode',
    choices=['sft', 'zero-shot', 'cross-lingual', 'instruct', 'instruct2'],
    default='sft',
    help='CosyVoice inference mode',
  )
  parser.add_argument('--host', default=_DEFAULT_HOST, help='Sidecar bind host')
  parser.add_argument(
    '--port', type=int, default=_DEFAULT_PORT, help='Sidecar bind port'
  )
  parser.add_argument(
    '--default-voice',
    default=None,
    help='Default speaker id for sft and instruct modes',
  )
  parser.add_argument(
    '--prompt-text',
    default=None,
    help='Prompt transcript for zero-shot mode',
  )
  parser.add_argument(
    '--prompt-wav',
    default=None,
    help='Prompt WAV path for zero-shot, cross-lingual, and instruct2',
  )
  parser.add_argument(
    '--instruct-text',
    default=None,
    help='Instruction text for instruct and instruct2 modes',
  )
  parser.add_argument(
    '--english-narration',
    action='store_true',
    help=(
      'Default English narration instruction for instruct and instruct2 when '
      '--instruct-text is omitted. Zero-shot still requires --prompt-text to '
      'match the English reference WAV transcript.'
    ),
  )
  parser.add_argument(
    '--sample-rate',
    type=int,
    default=_DEFAULT_SAMPLE_RATE,
    help='Sample rate advertised in stream response headers',
  )
  parser.add_argument(
    '--debug',
    action='store_true',
    help='Enable debug logging',
  )
  return parser.parse_args(argv)


def _require_nonblank(value: str | None, flag: str) -> str:
  if value is None or not value.strip():
    raise SystemExit(f'{flag} must not be empty')
  return value.strip()


def _resolve_existing_dir(path: str, flag: str) -> Path:
  stripped = path.strip()
  if not stripped:
    raise SystemExit(f'{flag} must not be empty')
  resolved = Path(stripped)
  if not resolved.is_dir():
    raise SystemExit(f'{flag} must be an existing directory: {stripped}')
  return resolved


def _resolve_prompt_wav(path: str | None, *, required_for: str) -> Path | None:
  if path is None:
    return None
  stripped = path.strip()
  if not stripped:
    raise SystemExit(f'--prompt-wav must not be empty for {required_for} mode')
  resolved = Path(stripped)
  if not resolved.is_file():
    raise SystemExit(f'--prompt-wav must be an existing readable file: {stripped}')
  try:
    with resolved.open('rb'):
      pass
  except OSError as exc:
    raise SystemExit(
      f'--prompt-wav must be an existing readable file: {stripped}'
    ) from exc
  return resolved


def _apply_english_narration_defaults(args: argparse.Namespace) -> None:
  if not args.english_narration:
    return
  if args.mode in ('instruct', 'instruct2') and not args.instruct_text:
    args.instruct_text = DEFAULT_ENGLISH_NARRATION_INSTRUCT


def settings_from_args(args: argparse.Namespace) -> LocalSidecarSettings:
  _apply_english_narration_defaults(args)
  if args.sample_rate <= 0:
    raise SystemExit('--sample-rate must be > 0')

  mode: SidecarMode = args.mode
  cosyvoice_repo = _resolve_existing_dir(args.cosyvoice_repo, '--cosyvoice-repo')
  model_dir = _resolve_existing_dir(args.model_dir, '--model-dir')
  default_voice = args.default_voice.strip() if args.default_voice else None
  prompt_text = args.prompt_text.strip() if args.prompt_text else None
  instruct_text = args.instruct_text.strip() if args.instruct_text else None
  prompt_wav_path: Path | None = None

  if mode == 'sft':
    if not default_voice:
      raise SystemExit('--default-voice is required for sft mode')
  elif mode == 'zero-shot':
    prompt_text = _require_nonblank(args.prompt_text, '--prompt-text')
    prompt_wav_path = _resolve_prompt_wav(args.prompt_wav, required_for='zero-shot')
    if prompt_wav_path is None:
      raise SystemExit('--prompt-wav is required for zero-shot mode')
  elif mode == 'cross-lingual':
    prompt_wav_path = _resolve_prompt_wav(args.prompt_wav, required_for='cross-lingual')
    if prompt_wav_path is None:
      raise SystemExit('--prompt-wav is required for cross-lingual mode')
  elif mode == 'instruct':
    if not default_voice:
      raise SystemExit('--default-voice is required for instruct mode')
    instruct_text = _require_nonblank(args.instruct_text, '--instruct-text')
  elif mode == 'instruct2':
    instruct_text = _require_nonblank(args.instruct_text, '--instruct-text')
    prompt_wav_path = _resolve_prompt_wav(args.prompt_wav, required_for='instruct2')
    if prompt_wav_path is None:
      raise SystemExit('--prompt-wav is required for instruct2 mode')

  return LocalSidecarSettings(
    mode=mode,
    cosyvoice_repo=cosyvoice_repo,
    model_dir=model_dir,
    default_voice=default_voice,
    prompt_text=prompt_text,
    prompt_wav_path=prompt_wav_path,
    instruct_text=instruct_text,
    sample_rate=args.sample_rate,
    channels=_DEFAULT_CHANNELS,
    pcm_format=_DEFAULT_PCM_FORMAT,
    sample_width=_DEFAULT_SAMPLE_WIDTH,
  )


def configure_cosyvoice_import_path(repo: Path) -> None:
  matcha = repo / 'third_party' / 'Matcha-TTS'
  for entry in (matcha, repo):
    path = str(entry.resolve())
    if path not in sys.path:
      sys.path.insert(0, path)


def load_automodel(settings: LocalSidecarSettings) -> Any:
  configure_cosyvoice_import_path(settings.cosyvoice_repo)
  from cosyvoice.cli.cosyvoice import AutoModel

  return AutoModel(model_dir=str(settings.model_dir))


def speech_to_pcm_bytes(speech: Any) -> bytes:
  if hasattr(speech, 'detach'):
    array = speech.detach().cpu().numpy()
  else:
    array = np.asarray(speech)
  clipped = np.clip(array, -1.0, 1.0)
  scaled = (clipped * (2**15 - 1)).astype('<i2')
  return scaled.tobytes()


def _resolve_voice(settings: LocalSidecarSettings, voice: str | None) -> str:
  selected = voice.strip() if voice else ''
  selected = selected or (settings.default_voice or '')
  return selected


def _dispatch_inference(
  settings: LocalSidecarSettings,
  model: Any,
  *,
  text: str,
  voice: str,
) -> Iterator[Mapping[str, Any]]:
  if settings.mode == 'sft':
    yield from model.inference_sft(text, voice, stream=True)
    return
  if settings.mode == 'zero-shot':
    assert settings.prompt_text is not None
    assert settings.prompt_wav_path is not None
    yield from model.inference_zero_shot(
      text,
      settings.prompt_text,
      str(settings.prompt_wav_path),
      stream=True,
    )
    return
  if settings.mode == 'cross-lingual':
    assert settings.prompt_wav_path is not None
    yield from model.inference_cross_lingual(
      text,
      str(settings.prompt_wav_path),
      stream=True,
    )
    return
  if settings.mode == 'instruct':
    assert settings.instruct_text is not None
    yield from model.inference_instruct(
      text,
      voice,
      settings.instruct_text,
      stream=True,
    )
    return
  assert settings.instruct_text is not None
  assert settings.prompt_wav_path is not None
  yield from model.inference_instruct2(
    text,
    settings.instruct_text,
    str(settings.prompt_wav_path),
    stream=True,
  )


def iter_model_pcm(
  settings: LocalSidecarSettings,
  model: Any,
  *,
  text: str,
  voice: str | None,
) -> Iterator[bytes]:
  resolved_voice = _resolve_voice(settings, voice)
  for item in _dispatch_inference(
    settings,
    model,
    text=text,
    voice=resolved_voice,
  ):
    yield speech_to_pcm_bytes(item['tts_speech'])


def _stream_response_headers(settings: LocalSidecarSettings) -> dict[str, str]:
  return {
    'Content-Type': 'audio/raw',
    'X-TTS-Sample-Rate': str(settings.sample_rate),
    'X-TTS-Channels': str(settings.channels),
    'X-TTS-Pcm-Format': settings.pcm_format,
    'X-TTS-Sample-Width': str(settings.sample_width),
    'X-TTS-Backend': _BACKEND_NAME,
  }


def health_payload(settings: LocalSidecarSettings) -> Mapping[str, object]:
  payload: dict[str, object] = {
    'status': 'ok',
    'backend': _BACKEND_NAME,
    'mode': settings.mode,
    'modelDir': str(settings.model_dir),
    'sampleRate': settings.sample_rate,
    'defaultVoiceConfigured': settings.default_voice is not None,
    'promptTextConfigured': settings.prompt_text is not None,
    'promptWavConfigured': settings.prompt_wav_path is not None,
    'instructTextConfigured': settings.instruct_text is not None,
  }
  if settings.prompt_wav_path is not None:
    payload['promptWavBasename'] = settings.prompt_wav_path.name
  return payload


def create_app(
  settings: LocalSidecarSettings,
  *,
  model_loader: Any | None = None,
) -> FastAPI:
  app = FastAPI(title='CosyVoice Local Sidecar')
  model_holder: dict[str, Any] = {'model': None}
  loader = model_loader or load_automodel

  def _get_model() -> Any:
    if model_holder['model'] is None:
      model_holder['model'] = loader(settings)
    return model_holder['model']

  @app.get('/health')
  async def health() -> JSONResponse:
    return JSONResponse(health_payload(settings))

  @app.post('/v1/tts/stream')
  async def stream_tts(body: TtsStreamRequest) -> StreamingResponse:
    model = _get_model()
    return StreamingResponse(
      iter_model_pcm(settings, model, text=body.text, voice=body.voice),
      media_type='audio/raw',
      headers=_stream_response_headers(settings),
    )

  return app


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(levelname)s %(message)s',
  )
  settings = settings_from_args(args)
  uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level='info')
  return 0


if __name__ == '__main__':
  sys.exit(main())
