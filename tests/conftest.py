from __future__ import annotations

from dataclasses import replace
from typing import TypedDict, Unpack

from tts_gateway.config import DeviceMode, EngineName, GatewayConfig, OutputFormat

_BASE_CONFIG = GatewayConfig(
  primary_engine='kokoro',
  fallback_engine=None,
  output_format='wav',
  chunk_max_chars=3000,
  request_timeout_seconds=3600,
  engine_timeout_seconds=30,
  ffmpeg_path='ffmpeg',
  kokoro_enabled=True,
  pocket_enabled=False,
  device_mode='cpu',
  models_dir='/tmp/models',
  default_voice=None,
  bind_host='127.0.0.1',
  bind_port=8000,
)


class _ConfigOverrides(TypedDict, total=False):
  primary_engine: EngineName
  fallback_engine: EngineName | None
  output_format: OutputFormat
  chunk_max_chars: int
  request_timeout_seconds: int
  engine_timeout_seconds: int
  ffmpeg_path: str
  kokoro_enabled: bool
  pocket_enabled: bool
  device_mode: DeviceMode
  models_dir: str
  default_voice: str | None
  bind_host: str
  bind_port: int


def _make_config(**overrides: Unpack[_ConfigOverrides]) -> GatewayConfig:
  return replace(_BASE_CONFIG, **overrides)
