"""Tests for engine resolution, chain behaviour, warmup, and ordering via JobRuntime."""

from __future__ import annotations

from dataclasses import replace

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
from tts_gateway.render import plan_chunks, synthesize_chunks
from tts_gateway.runtime import JobRuntime, NoEnginesError
from tts_gateway.types import SynthesisSpec


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


def _runtime(tmp_path, **overrides) -> JobRuntime:
  overrides.setdefault('data_dir', str(tmp_path / 'data'))
  return JobRuntime(_make_config(**overrides))


# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------


def test_enabled_selects_native(tmp_path) -> None:
  rt = _runtime(tmp_path, kokoro_enabled=True)
  info = rt.engine_info()
  assert info['kokoro']['mode'] == 'native'
  rt.close()


def test_disabled_engine_is_unavailable(tmp_path) -> None:
  rt = _runtime(tmp_path, kokoro_enabled=False)
  info = rt.engine_info()
  assert info['kokoro']['mode'] == 'disabled'
  rt.close()


# ---------------------------------------------------------------------------
# Chain behaviour — all engines unavailable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_unavailable_raises_error(tmp_path) -> None:
  rt = _runtime(tmp_path, kokoro_enabled=False, pocket_enabled=False)
  spec = rt.make_spec('hello')

  with pytest.raises(NoEnginesError):
    await rt.run_until_complete(spec)
  rt.close()


# ---------------------------------------------------------------------------
# Primary fails → fallback succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_fails_fallback_succeeds(tmp_path) -> None:
  rt = _runtime(
    tmp_path,
    kokoro_enabled=True,
    pocket_enabled=True,
    fallback_engine='pocket',
  )
  rt._engine_map['kokoro'] = FailingEngine('kokoro', EngineError('kokoro broke'))
  rt._engine_map['pocket'] = MockEngine('pocket')

  spec = rt.make_spec('hello')
  artifact = await rt.run_until_complete(spec)
  assert artifact.output_path.exists()
  rt.close()


@pytest.mark.asyncio
async def test_runtime_uses_config_default_voice(tmp_path) -> None:
  rt = _runtime(tmp_path, default_voice='af_bella')
  engine = MockEngine('kokoro')
  rt._engine_map['kokoro'] = engine

  spec = rt.make_spec('hello')
  assert spec.voice == 'af_bella'

  await rt.run_until_complete(spec)
  assert engine.calls[0] == ('hello', 'af_bella')
  rt.close()


# ---------------------------------------------------------------------------
# Engine info
# ---------------------------------------------------------------------------


def test_engine_info_native_and_disabled(tmp_path) -> None:
  rt = _runtime(tmp_path, kokoro_enabled=True, pocket_enabled=False)
  info = rt.engine_info()
  assert info['kokoro']['mode'] == 'native'
  assert info['pocket']['mode'] == 'disabled'
  rt.close()


def test_chunk_concurrency_uses_cpu_cap(
  tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr('tts_gateway.runtime.os.cpu_count', lambda: 8)
  rt = _runtime(tmp_path)
  assert rt.concurrency == 4
  rt.close()


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_loads_native_engines(tmp_path) -> None:
  rt = _runtime(tmp_path, kokoro_enabled=True, pocket_enabled=False)
  kokoro_engine = _WarmupEngine()
  rt._engine_map['kokoro'] = kokoro_engine

  results = await rt.warmup()
  assert 'kokoro' in results
  assert results['kokoro']['loaded'] is True
  assert results['kokoro']['device'] == 'cpu'
  assert kokoro_engine.load_count == 1
  rt.close()


# ---------------------------------------------------------------------------
# Per-engine timeout fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_timeout_falls_back(tmp_path) -> None:
  rt = _runtime(
    tmp_path,
    kokoro_enabled=True,
    pocket_enabled=True,
    fallback_engine='pocket',
    engine_timeout_seconds=1,
  )
  rt._engine_map['kokoro'] = SlowEngine('kokoro', delay=5.0)
  rt._engine_map['pocket'] = MockEngine('pocket')

  spec = rt.make_spec('hello')
  artifact = await rt.run_until_complete(spec)
  assert artifact.output_path.exists()
  rt.close()


# ---------------------------------------------------------------------------
# Concurrent chunk synthesis with ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_core_preserves_chunk_order() -> None:
  """Chunks must arrive in original text order despite parallel execution."""
  engine = StaggeredEngine(
    'kokoro',
    delays={
      'chunk-0': 0.05,
      'chunk-1': 0.0,
      'chunk-2': 0.02,
    },
  )

  plan = plan_chunks(
    SynthesisSpec(
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
