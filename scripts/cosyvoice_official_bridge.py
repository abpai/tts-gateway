#!/usr/bin/env python3
"""Adapt the official CosyVoice FastAPI runtime to the tts-gateway sidecar contract."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS_DIR))

from cosyvoice_defaults import (  # noqa: E402
  DEFAULT_ENGLISH_NARRATION_INSTRUCT,
)

_BRIDGE_NAME = 'cosyvoice-official'
_DEFAULT_UPSTREAM_BASE_URL = 'http://127.0.0.1:50001'
_DEFAULT_HOST = '127.0.0.1'
_DEFAULT_PORT = 50000
_DEFAULT_SAMPLE_RATE = 22050
_DEFAULT_CHANNELS = 1
_DEFAULT_PCM_FORMAT = 's16le'
_DEFAULT_SAMPLE_WIDTH = 2
_DEFAULT_REQUEST_TIMEOUT = 360.0
_UPSTREAM_PROBE_TIMEOUT = 2.0

BridgeMode = Literal['sft', 'zero-shot', 'cross-lingual', 'instruct', 'instruct2']

_MODE_ENDPOINTS: dict[BridgeMode, str] = {
  'sft': '/inference_sft',
  'zero-shot': '/inference_zero_shot',
  'cross-lingual': '/inference_cross_lingual',
  'instruct': '/inference_instruct',
  'instruct2': '/inference_instruct2',
}


@dataclass(frozen=True)
class BridgeSettings:
  mode: BridgeMode
  upstream_base_url: str
  upstream_endpoint: str
  default_voice: str | None
  prompt_text: str | None
  prompt_wav_path: Path | None
  instruct_text: str | None
  sample_rate: int
  channels: int
  pcm_format: str
  sample_width: int
  request_timeout: float


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
    '--mode',
    choices=list(_MODE_ENDPOINTS),
    default='sft',
    help='Official CosyVoice inference mode to adapt',
  )
  parser.add_argument(
    '--upstream-base-url',
    default=_DEFAULT_UPSTREAM_BASE_URL,
    help='Official CosyVoice FastAPI server base URL',
  )
  parser.add_argument(
    '--upstream-endpoint',
    default=None,
    help='Upstream inference endpoint path (defaults from --mode)',
  )
  parser.add_argument('--host', default=_DEFAULT_HOST, help='Bridge bind host')
  parser.add_argument(
    '--port', type=int, default=_DEFAULT_PORT, help='Bridge bind port'
  )
  parser.add_argument(
    '--default-voice',
    default=None,
    help='Default speaker id forwarded as upstream spk_id (sft, instruct)',
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
    '--request-timeout',
    type=float,
    default=_DEFAULT_REQUEST_TIMEOUT,
    help='Upstream request timeout in seconds',
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


def _resolve_upstream_endpoint(args: argparse.Namespace) -> str:
  if args.upstream_endpoint is not None:
    return _normalize_endpoint(args.upstream_endpoint)
  mode: BridgeMode = args.mode
  return _MODE_ENDPOINTS[mode]


def settings_from_args(args: argparse.Namespace) -> BridgeSettings:
  _apply_english_narration_defaults(args)
  if args.sample_rate <= 0:
    raise SystemExit('--sample-rate must be > 0')
  if args.request_timeout <= 0:
    raise SystemExit('--request-timeout must be > 0')

  mode: BridgeMode = args.mode
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

  return BridgeSettings(
    mode=mode,
    upstream_base_url=args.upstream_base_url.rstrip('/'),
    upstream_endpoint=_resolve_upstream_endpoint(args),
    default_voice=default_voice,
    prompt_text=prompt_text,
    prompt_wav_path=prompt_wav_path,
    instruct_text=instruct_text,
    sample_rate=args.sample_rate,
    channels=_DEFAULT_CHANNELS,
    pcm_format=_DEFAULT_PCM_FORMAT,
    sample_width=_DEFAULT_SAMPLE_WIDTH,
    request_timeout=args.request_timeout,
  )


def _normalize_endpoint(endpoint: str) -> str:
  normalized = endpoint.strip()
  if not normalized:
    raise SystemExit('--upstream-endpoint must not be empty')
  if normalized.startswith('/'):
    return normalized
  return f'/{normalized}'


def _upstream_url(settings: BridgeSettings) -> str:
  return f'{settings.upstream_base_url}{settings.upstream_endpoint}'


def _stream_response_headers(settings: BridgeSettings) -> dict[str, str]:
  return {
    'Content-Type': 'audio/raw',
    'X-TTS-Sample-Rate': str(settings.sample_rate),
    'X-TTS-Channels': str(settings.channels),
    'X-TTS-Pcm-Format': settings.pcm_format,
    'X-TTS-Sample-Width': str(settings.sample_width),
    'X-TTS-Upstream-Endpoint': settings.upstream_endpoint,
  }


def _resolve_voice(settings: BridgeSettings, voice: str | None) -> str:
  selected = voice.strip() if voice else ''
  selected = selected or (settings.default_voice or '')
  return selected


def _build_upstream_request(
  settings: BridgeSettings,
  *,
  text: str,
  voice: str,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]] | None]:
  if settings.mode == 'sft':
    return {'tts_text': text, 'spk_id': voice}, None
  if settings.mode == 'zero-shot':
    assert settings.prompt_text is not None
    assert settings.prompt_wav_path is not None
    return (
      {'tts_text': text, 'prompt_text': settings.prompt_text},
      _prompt_wav_files(settings.prompt_wav_path),
    )
  if settings.mode == 'cross-lingual':
    assert settings.prompt_wav_path is not None
    return {'tts_text': text}, _prompt_wav_files(settings.prompt_wav_path)
  if settings.mode == 'instruct':
    assert settings.instruct_text is not None
    return (
      {
        'tts_text': text,
        'spk_id': voice,
        'instruct_text': settings.instruct_text,
      },
      None,
    )
  assert settings.instruct_text is not None
  assert settings.prompt_wav_path is not None
  return (
    {'tts_text': text, 'instruct_text': settings.instruct_text},
    _prompt_wav_files(settings.prompt_wav_path),
  )


def _prompt_wav_files(
  path: Path,
) -> dict[str, tuple[str, bytes, str]]:
  return {'prompt_wav': (path.name, path.read_bytes(), 'audio/wav')}


async def _probe_upstream_reachable(settings: BridgeSettings) -> bool:
  probe_url = f'{settings.upstream_base_url}/docs'
  try:
    async with httpx.AsyncClient(timeout=_UPSTREAM_PROBE_TIMEOUT) as client:
      response = await client.get(probe_url)
      return 200 <= response.status_code < 500
  except httpx.RequestError:
    return False


async def _open_upstream_stream(
  settings: BridgeSettings,
  *,
  text: str,
  voice: str,
) -> tuple[httpx.AsyncClient, httpx.Response]:
  data, files = _build_upstream_request(settings, text=text, voice=voice)
  client = httpx.AsyncClient(timeout=settings.request_timeout)
  try:
    request = client.build_request(
      'POST',
      _upstream_url(settings),
      data=data,
      files=files,
    )
    response = await client.send(request, stream=True)
    if response.status_code < 200 or response.status_code >= 300:
      detail = (await response.aread()).decode('utf-8', errors='replace')
      await response.aclose()
      await client.aclose()
      raise HTTPException(
        status_code=response.status_code,
        detail=detail or 'upstream returned a non-success status',
      )
    return client, response
  except httpx.RequestError as exc:
    await client.aclose()
    raise HTTPException(
      status_code=502,
      detail=f'upstream request failed: {exc}',
    ) from exc


async def _stream_upstream_body(
  client: httpx.AsyncClient,
  response: httpx.Response,
) -> AsyncIterator[bytes]:
  try:
    async for chunk in response.aiter_bytes():
      yield chunk
  finally:
    await response.aclose()
    await client.aclose()


def _health_payload(
  settings: BridgeSettings,
  *,
  upstream_reachable: bool,
) -> Mapping[str, object]:
  payload: dict[str, object] = {
    'status': 'ok',
    'bridge': _BRIDGE_NAME,
    'mode': settings.mode,
    'upstreamBaseUrl': settings.upstream_base_url,
    'upstreamEndpoint': settings.upstream_endpoint,
    'sampleRate': settings.sample_rate,
    'channels': settings.channels,
    'pcmFormat': settings.pcm_format,
    'defaultVoice': settings.default_voice,
    'promptTextConfigured': settings.prompt_text is not None,
    'promptWavConfigured': settings.prompt_wav_path is not None,
    'instructTextConfigured': settings.instruct_text is not None,
    'upstreamReachable': upstream_reachable,
  }
  if settings.prompt_wav_path is not None:
    payload['promptWavBasename'] = settings.prompt_wav_path.name
  return payload


def create_app(settings: BridgeSettings) -> FastAPI:
  app = FastAPI(title='CosyVoice Official Bridge')

  @app.get('/health')
  async def health() -> JSONResponse:
    upstream_reachable = await _probe_upstream_reachable(settings)
    return JSONResponse(
      _health_payload(settings, upstream_reachable=upstream_reachable)
    )

  @app.post('/v1/tts/stream')
  async def stream_tts(body: TtsStreamRequest) -> StreamingResponse:
    voice = _resolve_voice(settings, body.voice)
    client, response = await _open_upstream_stream(
      settings,
      text=body.text,
      voice=voice,
    )
    return StreamingResponse(
      _stream_upstream_body(client, response),
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
