from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import TypedDict, Unpack

from tts_gateway.config import DeviceMode, EngineName, GatewayConfig, OutputFormat
from tts_gateway.engines.base import AudioChunk, Engine

_BASE_CONFIG = GatewayConfig(
  primary_engine='kokoro',
  fallback_engine=None,
  output_format='wav',
  chunk_max_chars=3000,
  stream_first_chunk_max_chars=180,
  stream_chunk_max_chars=3000,
  request_timeout_seconds=3600,
  engine_timeout_seconds=30,
  ffmpeg_path='ffmpeg',
  kokoro_enabled=True,
  pocket_enabled=False,
  cosyvoice_enabled=False,
  cosyvoice_base_url='http://127.0.0.1:50000',
  cosyvoice_request_timeout_seconds=30,
  device_mode='cpu',
  models_dir='/tmp/models',
  default_voice=None,
  bind_host='127.0.0.1',
  bind_port=8000,
  data_dir='/tmp/tts-data',
  pipeline_version='1',
  worker_poll_seconds=1.0,
)


class _ConfigOverrides(TypedDict, total=False):
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


def _make_config(**overrides: Unpack[_ConfigOverrides]) -> GatewayConfig:
  return replace(_BASE_CONFIG, **overrides)


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

DUMMY_CHUNK = AudioChunk(
  pcm_bytes=b'\x00\x01' * 100,
  sample_rate=24_000,
  channels=1,
  sample_width=2,
)


class MockEngine(Engine):
  """Simple engine that returns DUMMY_CHUNK."""

  def __init__(self, name: str = 'mock', chunk: AudioChunk = DUMMY_CHUNK) -> None:
    self.name = name
    self.chunk = chunk
    self.calls: list[tuple[str, str | None]] = []

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    self.calls.append((text, voice))
    return self.chunk


class FailingEngine(Engine):
  """Engine that always raises."""

  def __init__(self, name: str, error: Exception) -> None:
    self.name = name
    self._error = error

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    raise self._error


class SlowEngine(Engine):
  """Engine with a configurable delay."""

  def __init__(self, name: str, delay: float) -> None:
    self.name = name
    self._delay = delay

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    await asyncio.sleep(self._delay)
    return DUMMY_CHUNK


class StaggeredEngine(Engine):
  """Engine with per-text delays and concurrency tracking."""

  def __init__(self, name: str, delays: dict[str, float]) -> None:
    self.name = name
    self._delays = delays
    self.active_calls = 0
    self.max_active_calls = 0
    self.voices: list[str | None] = []

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    self.voices.append(voice)
    self.active_calls += 1
    self.max_active_calls = max(self.max_active_calls, self.active_calls)
    try:
      await asyncio.sleep(self._delays.get(text, 0))
      return AudioChunk(
        pcm_bytes=text.encode('utf-8'),
        sample_rate=24_000,
        channels=1,
        sample_width=2,
      )
    finally:
      self.active_calls -= 1


class MockStreamingEngine(Engine):
  """Engine that yields configured chunks via native streaming."""

  def __init__(
    self,
    name: str = 'stream-mock',
    chunks: tuple[AudioChunk, ...] | None = None,
  ) -> None:
    self.name = name
    self.chunks = chunks or (DUMMY_CHUNK,)
    self.stream_calls: list[tuple[str, str | None]] = []
    self.synth_calls: list[tuple[str, str | None]] = []

  async def stream_synthesize(
    self, text: str, *, voice: str | None = None
  ) -> AsyncGenerator[AudioChunk, None]:
    self.stream_calls.append((text, voice))
    for chunk in self.chunks:
      yield chunk

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    self.synth_calls.append((text, voice))
    return self.chunks[0]


class FailingStreamEngine(Engine):
  """Streaming engine that fails before or after yielding chunks."""

  def __init__(
    self,
    name: str,
    error: Exception,
    *,
    chunks_before_error: int = 0,
  ) -> None:
    self.name = name
    self._error = error
    self.chunks_before_error = chunks_before_error

  async def stream_synthesize(
    self, text: str, *, voice: str | None = None
  ) -> AsyncGenerator[AudioChunk, None]:
    for _ in range(self.chunks_before_error):
      yield DUMMY_CHUNK
    raise self._error

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    raise self._error


class SlowStreamEngine(Engine):
  """Streaming engine with a delay before the first chunk."""

  def __init__(self, name: str, delay: float) -> None:
    self.name = name
    self._delay = delay

  async def stream_synthesize(
    self, text: str, *, voice: str | None = None
  ) -> AsyncGenerator[AudioChunk, None]:
    await asyncio.sleep(self._delay)
    yield DUMMY_CHUNK

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    await asyncio.sleep(self._delay)
    return DUMMY_CHUNK


class SlowAfterFirstStreamEngine(Engine):
  """Streaming engine with a delay after the first chunk."""

  def __init__(self, name: str, delay: float) -> None:
    self.name = name
    self._delay = delay

  async def stream_synthesize(
    self, text: str, *, voice: str | None = None
  ) -> AsyncGenerator[AudioChunk, None]:
    yield DUMMY_CHUNK
    await asyncio.sleep(self._delay)
    yield DUMMY_CHUNK

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    await asyncio.sleep(self._delay)
    return DUMMY_CHUNK
