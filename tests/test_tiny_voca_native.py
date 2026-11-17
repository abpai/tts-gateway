from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.conftest import _make_config
from tts_gateway.engines.base import EngineError
from tts_gateway.engines.tiny_voca_native import TinyVocaNativeEngine


@pytest.fixture
def model_inputs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, np.ndarray]]:
  feeds: list[dict[str, np.ndarray]] = []

  def run(output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    assert output_names == ['audio']
    feeds.append(feed)
    sample = 2.0 if feed['tokens'][0, 0] == ord('b') else 0.5
    return [np.array([[sample]], dtype=np.float32)]

  def tokenize(pieces: list[tuple[bool, list[str]]]) -> SimpleNamespace:
    ids = tuple(map(ord, pieces[0][1][0]))
    return SimpleNamespace(ids=ids, stress=(1,) * len(ids))

  stubs = {
    'onnxruntime': SimpleNamespace(
      SessionOptions=SimpleNamespace,
      InferenceSession=lambda *args, **kwargs: SimpleNamespace(run=run),
    ),
    'dizione': SimpleNamespace(tokenize=tokenize),
    'dizione.engine': SimpleNamespace(
      Phonemizer=lambda path: SimpleNamespace(pieces=lambda text: [(False, [text])])
    ),
    'huggingface_hub': SimpleNamespace(
      hf_hub_download=lambda repo, filename, *, revision, cache_dir: (
        cache_dir / revision / filename
      )
    ),
  }
  for module, stub in stubs.items():
    monkeypatch.setitem(sys.modules, module, stub)
  return feeds


@pytest.fixture
def engine(tmp_path: Path) -> TinyVocaNativeEngine:
  return TinyVocaNativeEngine(
    _make_config(tiny_voca_enabled=True, models_dir=str(tmp_path))
  )


@pytest.mark.parametrize(
  'voice,speed,voice_id',
  [(None, 1.0, 0), ('F2', 0.75, 0), ('M2', 1.5, 1)],
)
async def test_synthesize_uses_selected_voice_and_native_speed(
  engine: TinyVocaNativeEngine,
  model_inputs: list[dict[str, np.ndarray]],
  voice: str | None,
  speed: float,
  voice_id: int,
) -> None:
  audio = await engine.synthesize('Hello', voice=voice, speed=speed)

  assert (audio.sample_rate, audio.channels, audio.sample_width) == (24_000, 1, 2)
  assert audio.speed_applied == (None if speed == 1.0 else speed)
  assert np.frombuffer(audio.pcm_bytes, dtype=np.int16).tolist() == [16_383]
  (feed,) = model_inputs
  expected: dict[str, NDArray] = {
    'tokens': np.array([list(map(ord, 'Hello.'))], dtype=np.int64),
    'stress': np.array([[1] * 6], dtype=np.int64),
    'voice': np.array([voice_id], dtype=np.int64),
    'speed': np.array([speed], dtype=np.float32),
    'pitch_shift': np.array([0.0], dtype=np.float32),
    'pitch_range': np.array([1.0], dtype=np.float32),
  }
  assert feed.keys() == expected.keys()
  for name, values in expected.items():
    np.testing.assert_array_equal(feed[name], values, strict=True)


async def test_synthesize_splits_long_text_and_keeps_audio_in_order(
  engine: TinyVocaNativeEngine, model_inputs: list[dict[str, np.ndarray]]
) -> None:
  first, second = 'a' * 300, 'b' * 300
  audio = await engine.synthesize(f'{first}, {second}')

  assert len(model_inputs) == 2
  tokens = [feed['tokens'][0] for feed in model_inputs]
  assert all(len(utterance) <= 512 for utterance in tokens)
  spoken = ''.join(chr(token) for utterance in tokens for token in utterance)
  assert ''.join(filter(str.isalpha, spoken)) == first + second
  quiet, loud = np.frombuffer(audio.pcm_bytes, dtype=np.int16)
  assert 0 < quiet < loud < 32767


@pytest.mark.parametrize(
  'module,hint',
  [
    ('onnxruntime', r'onnxruntime.*--extra tiny-voca'),
    ('dizione', r'Dizione.*github.com/n4ze3m/dizione'),
  ],
)
async def test_synthesize_reports_missing_optional_dependency(
  engine: TinyVocaNativeEngine,
  model_inputs: list[dict[str, np.ndarray]],
  monkeypatch: pytest.MonkeyPatch,
  module: str,
  hint: str,
) -> None:
  monkeypatch.setitem(sys.modules, module, None)

  with pytest.raises(EngineError, match=hint):
    await engine.synthesize('Hello')

  assert model_inputs == []
