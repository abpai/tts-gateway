"""Tests for LazyNativeEngine base class behaviour.

All tests use a fake subclass — no real model files or heavy packages required.
"""

from __future__ import annotations

import asyncio

import pytest

from tts_gateway.engines.base import AudioChunk, EngineError
from tts_gateway.engines.native_engine import LazyNativeEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_CHUNK = AudioChunk(
  pcm_bytes=b'\x00\x01' * 100,
  sample_rate=24_000,
  channels=1,
  sample_width=2,
)


class _FakeNativeEngine(LazyNativeEngine):
  """Minimal concrete subclass for testing the base class."""

  def __init__(
    self,
    *,
    enabled: bool = True,
    load_raises: Exception | None = None,
    inference_result: AudioChunk = _DUMMY_CHUNK,
    required_module: str | None = None,
    install_hint: str | None = None,
  ) -> None:
    super().__init__(
      'fake',
      enabled=enabled,
      models_dir='/tmp/models',
      device_mode='cpu',
    )
    self._load_raises = load_raises
    self._inference_result = inference_result
    self._required_module = required_module
    self._install_hint = install_hint
    self.load_count = 0
    self.inference_count = 0

  def _load_model(self) -> None:
    self.load_count += 1
    if self._load_raises:
      raise self._load_raises
    self._device = 'cpu'

  def _run_inference(self, text: str, voice: str | None = None) -> AudioChunk:
    self.inference_count += 1
    return self._inference_result

  def required_module_name(self) -> str | None:
    return self._required_module

  def install_hint(self) -> str | None:
    return self._install_hint


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_call_loads_model() -> None:
  engine = _FakeNativeEngine()
  assert not engine._loaded
  await engine.synthesize('hello')
  assert engine._loaded
  assert engine.load_count == 1


@pytest.mark.asyncio
async def test_subsequent_calls_skip_load() -> None:
  engine = _FakeNativeEngine()
  await engine.synthesize('first')
  await engine.synthesize('second')
  assert engine.load_count == 1
  assert engine.inference_count == 2


@pytest.mark.asyncio
async def test_concurrent_calls_load_once() -> None:
  engine = _FakeNativeEngine()

  async def call() -> AudioChunk:
    return await engine.synthesize('concurrent')

  results = await asyncio.gather(call(), call(), call())
  assert len(results) == 3
  assert engine.load_count == 1


# ---------------------------------------------------------------------------
# Disabled engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_engine_raises() -> None:
  engine = _FakeNativeEngine(enabled=False)
  with pytest.raises(EngineError, match='disabled'):
    await engine.synthesize('should fail')
  assert engine.load_count == 0


# ---------------------------------------------------------------------------
# Load failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_failure_raises_engine_error() -> None:
  engine = _FakeNativeEngine(load_raises=RuntimeError('bad model'))
  with pytest.raises(EngineError, match='failed to load'):
    await engine.synthesize('should fail')
  assert engine._load_error is not None


@pytest.mark.asyncio
async def test_load_failure_retries_on_next_call() -> None:
  """After a failed load, subsequent calls should re-attempt the load."""
  engine = _FakeNativeEngine(load_raises=RuntimeError('transient'))

  with pytest.raises(EngineError):
    await engine.synthesize('attempt 1')

  # Fix the load and try again
  engine._load_raises = None
  await engine.synthesize('attempt 2')
  assert engine._loaded
  assert engine.load_count == 2
  assert engine.health_status()['loadError'] is None


@pytest.mark.asyncio
async def test_missing_dependency_reports_install_hint() -> None:
  engine = _FakeNativeEngine(
    load_raises=ModuleNotFoundError("No module named 'missing_pkg'"),
    required_module='missing_pkg',
    install_hint='uv sync --group dev --extra missing',
  )

  with pytest.raises(
    EngineError,
    match='install it with: uv sync --group dev --extra missing',
  ):
    await engine.synthesize('should fail')

  assert (
    engine.health_status()['loadError']
    == 'missing dependency "missing_pkg"; install it with: uv sync --group dev --extra missing'
  )


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------


def test_health_status_initial() -> None:
  engine = _FakeNativeEngine()
  status = engine.health_status()
  assert status == {
    'mode': 'native',
    'enabled': True,
    'loaded': False,
    'device': 'unknown',
    'loadError': None,
  }


@pytest.mark.asyncio
async def test_health_status_after_load() -> None:
  engine = _FakeNativeEngine()
  await engine.synthesize('hello')
  status = engine.health_status()
  assert status['loaded'] is True
  assert status['device'] == 'cpu'


@pytest.mark.asyncio
async def test_health_status_after_failure() -> None:
  engine = _FakeNativeEngine(load_raises=RuntimeError('boom'))
  with pytest.raises(EngineError):
    await engine.synthesize('fail')

  status = engine.health_status()
  assert status['loaded'] is False
  assert status['loadError'] == 'boom'


@pytest.mark.asyncio
async def test_health_status_clears_load_error_after_retry_success() -> None:
  engine = _FakeNativeEngine(load_raises=RuntimeError('transient'))

  with pytest.raises(EngineError):
    await engine.synthesize('fail')

  engine._load_raises = None
  await engine.ensure_loaded()

  status = engine.health_status()
  assert status['loaded'] is True
  assert status['loadError'] is None


# ---------------------------------------------------------------------------
# ensure_loaded (public API for /warmup)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_loaded_public() -> None:
  engine = _FakeNativeEngine()
  await engine.ensure_loaded()
  assert engine._loaded
  assert engine.load_count == 1
  # Calling again is a no-op
  await engine.ensure_loaded()
  assert engine.load_count == 1
