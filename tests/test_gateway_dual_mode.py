"""Tests for gateway dual-mode engine resolution and chain behaviour."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from tests.conftest import _make_config
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


class _StaggeredEngine(TtsEngine):
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
      await asyncio.sleep(self._delays[text])
      return AudioChunk(
        pcm_bytes=text.encode('utf-8'),
        sample_rate=24_000,
        channels=1,
        sample_width=2,
      )
    finally:
      self.active_calls -= 1


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


def test_chunk_concurrency_uses_cpu_cap(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr('tts_gateway.gateway.os.cpu_count', lambda: 8)
  gw = TtsGateway(_make_config())
  assert gw.chunk_concurrency() == 4


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


@pytest.mark.asyncio
async def test_audio_chunks_run_concurrently_and_preserve_order() -> None:
  cfg = _make_config()
  gw = TtsGateway(cfg)
  engine = _StaggeredEngine(
    'kokoro',
    delays={
      'chunk-0': 0.05,
      'chunk-1': 0.0,
      'chunk-2': 0.02,
    },
  )
  gw.engines['kokoro'] = engine
  attempts: list = []

  audio_chunks = await gw._synthesize_audio_chunks(
    ['chunk-0', 'chunk-1', 'chunk-2'],
    attempts,
    voice='af_bella',
  )

  assert engine.max_active_calls >= 2
  assert engine.voices == ['af_bella', 'af_bella', 'af_bella']
  assert [chunk.pcm_bytes for chunk in audio_chunks] == [
    b'chunk-0',
    b'chunk-1',
    b'chunk-2',
  ]
  assert [(attempt.chunk_index, attempt.engine) for attempt in attempts] == [
    (0, 'kokoro'),
    (1, 'kokoro'),
    (2, 'kokoro'),
  ]
