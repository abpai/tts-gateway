from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.conftest import _make_config
from tts_gateway.engines import kokoro_native
from tts_gateway.engines.base import EngineError
from tts_gateway.engines.kokoro_native import KokoroNativeEngine


def _fake_torch(*, cuda_available: bool, mps_available: bool) -> SimpleNamespace:
  return SimpleNamespace(
    cuda=SimpleNamespace(is_available=lambda: cuda_available),
    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
  )


def test_resolve_device_cpu_mode() -> None:
  assert KokoroNativeEngine._resolve_device('cpu') == 'cpu'


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=True, mps_available=True)
  )
  assert KokoroNativeEngine._resolve_device('auto') == 'cuda'


def test_resolve_device_auto_uses_cpu_when_cuda_missing(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=False, mps_available=True)
  )
  assert KokoroNativeEngine._resolve_device('auto') == 'cpu'


def test_resolve_device_mps_mode_still_opts_into_mps(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=False, mps_available=True)
  )
  assert KokoroNativeEngine._resolve_device('mps') == 'mps'


def test_resolve_device_cuda_mode_falls_back_when_unavailable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=False, mps_available=True)
  )
  assert KokoroNativeEngine._resolve_device('cuda') == 'cpu'


def test_resolve_device_mps_mode_falls_back_when_unavailable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=True, mps_available=False)
  )
  assert KokoroNativeEngine._resolve_device('mps') == 'cpu'


class _FakePipeline:
  def __init__(self, segments: list[Any], error: Exception | None = None) -> None:
    self.segments = segments
    self.error = error
    self.calls: list[tuple[str, str, float]] = []
    self.split_patterns: list[str] = []
    self.voices = {}

  def __call__(self, text: str, *, voice: str, speed: float, split_pattern: str):
    self.calls.append((text, voice, speed))
    self.split_patterns.append(split_pattern)
    for segment in self.segments:
      yield None, None, segment
    if self.error is not None:
      raise self.error


def _loaded_engine(pipeline: _FakePipeline) -> KokoroNativeEngine:
  engine = KokoroNativeEngine(_make_config())
  engine._loaded = True
  engine._pipeline = pipeline
  return engine


@pytest.mark.asyncio
async def test_stream_synthesize_yields_pipeline_segments_in_order() -> None:
  pipeline = _FakePipeline(
    [
      np.array([0.0, 0.25], dtype=np.float32),
      np.array([0.5], dtype=np.float32),
    ]
  )
  engine = _loaded_engine(pipeline)

  chunks = [
    chunk
    async for chunk in engine.stream_synthesize(
      'Hello. Bye.',
      voice='af_test',
      speed=1.5,
    )
  ]

  assert len(chunks) == 2
  assert chunks[0].sample_rate == 24_000
  assert chunks[0].channels == 1
  assert chunks[0].sample_width == 2
  assert chunks[0].speed_applied == 1.5
  assert pipeline.calls == [('Hello. Bye.', 'af_test', 1.5)]
  # The pipeline must segment on sentence boundaries, not only newlines,
  # so streamed audio starts after the first sentence of a paragraph.
  assert pipeline.split_patterns == [kokoro_native._SEGMENT_SPLIT_PATTERN]


def test_segment_split_pattern_splits_sentences_and_newlines() -> None:
  parts = re.split(
    kokoro_native._SEGMENT_SPLIT_PATTERN,
    'First sentence. Second one!\nThird line',
  )
  assert parts == ['First sentence.', 'Second one!', 'Third line']


@pytest.mark.asyncio
async def test_stream_synthesize_empty_output_raises() -> None:
  engine = _loaded_engine(_FakePipeline([]))

  with pytest.raises(EngineError, match='kokoro produced no audio output'):
    async for _ in engine.stream_synthesize('Hello'):
      pass


def test_require_spacy_model_passes_when_installed(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'spacy.util', SimpleNamespace(is_package=lambda name: True)
  )

  kokoro_native.KokoroNativeEngine._require_spacy_model()


def test_require_spacy_model_fails_fast_with_install_hint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'spacy.util', SimpleNamespace(is_package=lambda name: False)
  )

  with pytest.raises(RuntimeError, match=r'en_core_web_sm.*uv'):
    kokoro_native.KokoroNativeEngine._require_spacy_model()
