"""Tests for gateway dual-mode engine resolution and chain behaviour."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.conftest import (
  DUMMY_CHUNK,
  FailingEngine,
  MockEngine,
  SlowEngine,
  StaggeredEngine,
  _make_config,
)
from tts_gateway.engines.base import AudioChunk, EngineError
from tts_gateway.engines.native_engine import LazyNativeEngine
from tts_gateway.gateway import SynthesisError, TtsGateway


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
    return DUMMY_CHUNK


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

  gw.engines['kokoro'] = FailingEngine('kokoro', EngineError('kokoro broke'))
  gw.engines['pocket'] = MockEngine('pocket')

  with patch('tts_gateway.gateway.encode_output', return_value=(b'audio', 'audio/wav')):
    result = await gw.synthesize('hello')

  assert result.payload == b'audio'


@pytest.mark.asyncio
async def test_gateway_uses_config_default_voice() -> None:
  cfg = _make_config(default_voice='af_bella')
  gw = TtsGateway(cfg)
  engine = MockEngine('kokoro')
  gw.engines['kokoro'] = engine

  with patch('tts_gateway.gateway.encode_output', return_value=(b'audio', 'audio/wav')):
    result = await gw.synthesize('hello')

  assert result.payload == b'audio'
  assert engine.calls == [('hello', 'af_bella')]


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
async def test_timeout_raises_synthesis_error() -> None:
  """Request timeout should raise SynthesisError with timed_out flag."""
  cfg = _make_config(
    kokoro_enabled=True,
    pocket_enabled=True,
    fallback_engine='pocket',
    request_timeout_seconds=1,
  )
  gw = TtsGateway(cfg)
  gw.engines['kokoro'] = SlowEngine('kokoro', delay=5.0)
  gw.engines['pocket'] = SlowEngine('pocket', delay=5.0)

  with pytest.raises(SynthesisError) as exc_info:
    await gw.synthesize_with_timeout('hello')

  assert exc_info.value.timed_out is True


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
  gw.engines['kokoro'] = SlowEngine('kokoro', delay=5.0)
  gw.engines['pocket'] = MockEngine('pocket')

  with patch('tts_gateway.gateway.encode_output', return_value=(b'audio', 'audio/wav')):
    result = await gw.synthesize('hello')

  assert result.payload == b'audio'


# ---------------------------------------------------------------------------
# Concurrent chunk synthesis with ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_core_preserves_chunk_order() -> None:
  """Chunks must arrive in original text order despite parallel execution."""
  from tts_gateway.synthesis import SynthesisRequest, plan_chunks, synthesize_chunks

  engine = StaggeredEngine(
    'kokoro',
    delays={
      'chunk-0': 0.05,
      'chunk-1': 0.0,
      'chunk-2': 0.02,
    },
  )

  plan = plan_chunks(
    SynthesisRequest(
      text='chunk-0',
      voice='af_bella',
      output_format='wav',
      chunk_max_chars=3000,
    )
  )
  plan = replace(plan, chunks=('chunk-0', 'chunk-1', 'chunk-2'))

  audio_chunks = []
  async for chunk in synthesize_chunks(
    plan, [engine], concurrency=4, engine_timeout=10
  ):
    audio_chunks.append(chunk)

  assert engine.max_active_calls >= 2
  assert engine.voices == ['af_bella', 'af_bella', 'af_bella']
  assert [chunk.pcm_bytes for chunk in audio_chunks] == [
    b'chunk-0',
    b'chunk-1',
    b'chunk-2',
  ]
