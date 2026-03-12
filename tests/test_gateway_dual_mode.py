"""Tests for gateway dual-mode engine resolution and chain behaviour."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TypedDict, Unpack
from unittest.mock import patch

import pytest

from tts_gateway.config import DeviceMode, EngineName, GatewayConfig, OutputFormat
from tts_gateway.engines.base import AudioChunk, EngineError, TtsEngine
from tts_gateway.engines.native_engine import LazyNativeEngine
from tts_gateway.gateway import SynthesisError, TtsGateway

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DUMMY_CHUNK = AudioChunk(
  pcm_bytes=b'\x00\x01' * 100,
  sample_rate=24_000,
  channels=1,
  sample_width=2,
)

_BASE_CONFIG = GatewayConfig(
  primary_engine='kokoro',
  fallback_engine=None,
  output_format='wav',
  chunk_max_chars=1400,
  request_timeout_seconds=60,
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


class _FailingEngine(TtsEngine):
  def __init__(self, name: str, error: Exception) -> None:
    self.name = name
    self._error = error

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    raise self._error


class _RecordingEngine(TtsEngine):
  def __init__(self, name: str, chunk: AudioChunk = _DUMMY_CHUNK) -> None:
    self.name = name
    self.chunk = chunk
    self.voices: list[str | None] = []

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    self.voices.append(voice)
    return self.chunk


class _SlowEngine(TtsEngine):
  def __init__(self, name: str, delay: float) -> None:
    self.name = name
    self._delay = delay

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    await asyncio.sleep(self._delay)
    return _DUMMY_CHUNK


class _WarmupEngine(LazyNativeEngine):
  def __init__(self) -> None:
    super().__init__(
      'kokoro',
      enabled=True,
      models_dir='/tmp/models',
      device_mode='cpu',
    )
    self.load_count = 0

  def _load_model(self) -> None:
    self.load_count += 1
    self._device = 'cpu'

  def _run_inference(self, text: str, voice: str | None = None) -> AudioChunk:
    return _DUMMY_CHUNK


def _make_config(**overrides: Unpack[_ConfigOverrides]) -> GatewayConfig:
  return replace(_BASE_CONFIG, **overrides)


# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------


def test_enabled_selects_native() -> None:
  """When engine is enabled, use native mode."""
  cfg = _make_config(kokoro_enabled=True)
  gw = TtsGateway(cfg)
  info = gw.engine_info()
  assert info['kokoro']['mode'] == 'native'


def test_disabled_engine_is_unavailable() -> None:
  """When engine is disabled, it is unavailable."""
  cfg = _make_config(kokoro_enabled=False)
  gw = TtsGateway(cfg)
  info = gw.engine_info()
  assert info['kokoro']['mode'] == 'disabled'


# ---------------------------------------------------------------------------
# Chain behaviour — all engines unavailable → 503
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_unavailable_raises_503() -> None:
  cfg = _make_config(
    kokoro_enabled=False,
    pocket_enabled=False,
  )
  gw = TtsGateway(cfg)

  with pytest.raises(SynthesisError) as exc_info:
    await gw.synthesize('hello')

  assert exc_info.value.unavailable is True
  assert len(exc_info.value.attempts) >= 1
  assert all(not a.ok for a in exc_info.value.attempts)


# ---------------------------------------------------------------------------
# Primary fails → fallback succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_fails_fallback_succeeds() -> None:
  cfg = _make_config(
    kokoro_enabled=True,
    pocket_enabled=True,
    fallback_engine='pocket',
  )
  gw = TtsGateway(cfg)

  gw.engines['kokoro'] = _FailingEngine('kokoro', EngineError('kokoro broke'))
  gw.engines['pocket'] = _RecordingEngine('pocket')

  with patch('tts_gateway.gateway.encode_output', return_value=(b'audio', 'audio/wav')):
    result = await gw.synthesize('hello')

  assert result.payload == b'audio'
  assert len(result.attempts) == 2
  assert result.attempts[0].engine == 'kokoro'
  assert result.attempts[0].ok is False
  assert result.attempts[1].engine == 'pocket'
  assert result.attempts[1].ok is True


@pytest.mark.asyncio
async def test_gateway_uses_config_default_voice() -> None:
  cfg = _make_config(default_voice='af_bella')
  gw = TtsGateway(cfg)
  engine = _RecordingEngine('kokoro')
  gw.engines['kokoro'] = engine

  with patch('tts_gateway.gateway.encode_output', return_value=(b'audio', 'audio/wav')):
    result = await gw.synthesize('hello')

  assert result.payload == b'audio'
  assert engine.voices == ['af_bella']


# ---------------------------------------------------------------------------
# Engine info
# ---------------------------------------------------------------------------


def test_engine_info_native_and_disabled() -> None:
  cfg = _make_config(
    kokoro_enabled=True,
    pocket_enabled=False,
  )
  gw = TtsGateway(cfg)
  info = gw.engine_info()
  assert info['kokoro']['mode'] == 'native'
  assert info['pocket']['mode'] == 'disabled'


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_loads_native_engines() -> None:
  cfg = _make_config(kokoro_enabled=True, pocket_enabled=False)
  gw = TtsGateway(cfg)
  kokoro_engine = _WarmupEngine()
  gw.engines['kokoro'] = kokoro_engine

  results = await gw.warmup()
  assert 'kokoro' in results
  assert results['kokoro']['loaded'] is True
  assert results['kokoro']['device'] == 'cpu'
  assert kokoro_engine.load_count == 1


# ---------------------------------------------------------------------------
# Timeout preserves attempt history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_preserves_attempts() -> None:
  """Request timeout should include attempts completed before the timeout."""
  cfg = _make_config(
    kokoro_enabled=True,
    pocket_enabled=True,
    fallback_engine='pocket',
    request_timeout_seconds=1,
  )
  gw = TtsGateway(cfg)
  gw.engines['kokoro'] = _FailingEngine('kokoro', EngineError('kokoro broke'))
  gw.engines['pocket'] = _SlowEngine('pocket', delay=5.0)

  with pytest.raises(SynthesisError) as exc_info:
    await gw.synthesize_with_timeout('hello')

  assert exc_info.value.timed_out is True
  assert len(exc_info.value.attempts) >= 1
  assert exc_info.value.attempts[0].engine == 'kokoro'
  assert exc_info.value.attempts[0].ok is False


# ---------------------------------------------------------------------------
# Per-engine timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_timeout_falls_back() -> None:
  """An engine that exceeds engine_timeout_seconds should be skipped."""
  cfg = _make_config(
    kokoro_enabled=True,
    pocket_enabled=True,
    fallback_engine='pocket',
    engine_timeout_seconds=1,
  )
  gw = TtsGateway(cfg)
  gw.engines['kokoro'] = _SlowEngine('kokoro', delay=5.0)
  gw.engines['pocket'] = _RecordingEngine('pocket')

  with patch('tts_gateway.gateway.encode_output', return_value=(b'audio', 'audio/wav')):
    result = await gw.synthesize('hello')

  assert result.payload == b'audio'
  assert len(result.attempts) == 2
  assert result.attempts[0].engine == 'kokoro'
  assert result.attempts[0].ok is False
  assert 'timed out' in (result.attempts[0].error or '')
  assert result.attempts[1].engine == 'pocket'
  assert result.attempts[1].ok is True
