from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

EngineName = Literal['kokoro', 'pocket', 'cosyvoice']
EngineSetting = EngineName | Literal['none']
OutputFormat = Literal['mp3', 'wav']
DeviceMode = Literal['auto', 'cuda', 'mps', 'cpu']

DEFAULT_MODELS_DIR = '~/.cache/tts-gateway/models'
DEFAULT_DATA_DIR = '~/.cache/tts-gateway/data'


@dataclass(frozen=True)
class GatewayConfig:
  primary_engine: EngineName
  fallback_engine: EngineName | None
  output_format: OutputFormat
  chunk_max_chars: int
  stream_first_chunk_max_chars: int
  stream_chunk_max_chars: int
  request_timeout_seconds: int
  engine_timeout_seconds: int
  ffmpeg_path: str
  kokoro_enabled: bool
  pocket_enabled: bool
  cosyvoice_enabled: bool
  cosyvoice_base_url: str
  cosyvoice_request_timeout_seconds: int
  device_mode: DeviceMode
  models_dir: str
  default_voice: str | None
  bind_host: str
  bind_port: int
  data_dir: str
  pipeline_version: str
  worker_poll_seconds: float


def _optional(name: str, default: str) -> str:
  value = os.getenv(name, default).strip()
  return value if value else default


def _parse_positive_int(name: str, default: int) -> int:
  raw = _optional(name, str(default))
  try:
    parsed = int(raw)
  except ValueError as exc:
    raise ValueError(f'{name} must be an integer, received: {raw}') from exc

  if parsed <= 0:
    raise ValueError(f'{name} must be > 0, received: {raw}')

  return parsed


def _parse_bool(name: str, default: bool) -> bool:
  raw = os.getenv(name, str(default)).strip().lower()
  if raw in {'true', '1', 'yes', 'on'}:
    return True
  if raw in {'false', '0', 'no', 'off', ''}:
    return False
  raise ValueError(f'{name} must be a boolean (true/false), received: {raw}')


def _parse_engine(
  name: str, default: EngineSetting, allow_none: bool
) -> EngineName | None:
  raw = _optional(name, default).lower()
  if allow_none and raw in {'', 'none', 'off'}:
    return None
  if raw == 'kokoro':
    return 'kokoro'
  if raw == 'pocket':
    return 'pocket'
  if raw == 'cosyvoice':
    return 'cosyvoice'
  raise ValueError(
    f'{name} must be one of kokoro, pocket, cosyvoice, none; received: {raw}'
  )


def _parse_output_format(name: str, default: OutputFormat) -> OutputFormat:
  raw = _optional(name, default).lower()
  if raw == 'mp3':
    return 'mp3'
  if raw == 'wav':
    return 'wav'
  raise ValueError(f'{name} must be one of mp3, wav; received: {raw}')


def _parse_device_mode(name: str, default: DeviceMode) -> DeviceMode:
  raw = _optional(name, default).lower()
  if raw in {'auto', 'cuda', 'mps', 'cpu'}:
    return raw  # type: ignore[return-value]
  raise ValueError(f'{name} must be one of auto, cuda, mps, cpu; received: {raw}')


def _parse_optional_str(name: str) -> str | None:
  value = os.getenv(name, '').strip()
  return value or None


def load_config() -> GatewayConfig:
  primary_engine = _parse_engine('TTS_PRIMARY_ENGINE', 'kokoro', allow_none=False)
  if primary_engine is None:
    raise ValueError('TTS_PRIMARY_ENGINE cannot be none')

  fallback_engine = _parse_engine('TTS_FALLBACK_ENGINE', 'none', allow_none=True)

  chunk_max_chars = _parse_positive_int('TTS_CHUNK_MAX_CHARS', 500)
  engine_timeout_seconds = _parse_positive_int('TTS_ENGINE_TIMEOUT_SECONDS', 360)
  cosyvoice_timeout_raw = os.getenv('TTS_COSYVOICE_REQUEST_TIMEOUT_SECONDS', '').strip()
  cosyvoice_request_timeout_seconds = (
    _parse_positive_int('TTS_COSYVOICE_REQUEST_TIMEOUT_SECONDS', engine_timeout_seconds)
    if cosyvoice_timeout_raw
    else engine_timeout_seconds
  )

  return GatewayConfig(
    primary_engine=primary_engine,
    fallback_engine=fallback_engine,
    output_format=_parse_output_format('TTS_OUTPUT_FORMAT', 'mp3'),
    chunk_max_chars=chunk_max_chars,
    stream_first_chunk_max_chars=_parse_positive_int(
      'TTS_STREAM_FIRST_CHUNK_MAX_CHARS', 180
    ),
    stream_chunk_max_chars=_parse_positive_int(
      'TTS_STREAM_CHUNK_MAX_CHARS', chunk_max_chars
    ),
    request_timeout_seconds=_parse_positive_int('TTS_REQUEST_TIMEOUT_SECONDS', 3600),
    engine_timeout_seconds=engine_timeout_seconds,
    ffmpeg_path=_optional('TTS_FFMPEG_PATH', 'ffmpeg'),
    kokoro_enabled=_parse_bool('KOKORO_TTS_ENABLED', True),
    pocket_enabled=_parse_bool('POCKET_TTS_ENABLED', False),
    cosyvoice_enabled=_parse_bool('COSYVOICE_TTS_ENABLED', False),
    cosyvoice_base_url=_optional('TTS_COSYVOICE_BASE_URL', 'http://127.0.0.1:50000'),
    cosyvoice_request_timeout_seconds=cosyvoice_request_timeout_seconds,
    device_mode=_parse_device_mode('TTS_DEVICE_MODE', 'auto'),
    models_dir=_optional('TTS_MODELS_DIR', os.path.expanduser(DEFAULT_MODELS_DIR)),
    default_voice=_parse_optional_str('TTS_DEFAULT_VOICE'),
    bind_host=_optional('TTS_GATEWAY_HOST', '127.0.0.1'),
    bind_port=_parse_positive_int('TTS_GATEWAY_PORT', 8000),
    data_dir=_optional('TTS_DATA_DIR', os.path.expanduser(DEFAULT_DATA_DIR)),
    pipeline_version=_optional('TTS_PIPELINE_VERSION', '1'),
    worker_poll_seconds=float(_optional('TTS_WORKER_POLL_SECONDS', '1.0')),
  )
