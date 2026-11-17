from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from tts_gateway.chunking import SENTENCE_BOUNDARY
from tts_gateway.config import GatewayConfig
from tts_gateway.engines.base import AudioChunk, EngineError
from tts_gateway.engines.native_engine import LazyNativeEngine

_REPO = 'n4ze3m/tiny-voca'
_REVISION = 'e5f9b11651a4db525fb207a56a91e4df2d1dd387'
_DIZIONE_SOURCE = (
  'git+https://github.com/n4ze3m/dizione@8cd889e4de7be6bd1dd2e6891344a0729fa14dd4'
)
_PHONEMIZER_FILES = (
  'lexicon.tsv',
  'align.json',
  'g2p.json',
  'g2p.npz',
  'homographs.json',
)
_VOICES = {'F2': 0, 'M2': 1}
_MAX_PHONEMES = 500  # 512 model tokens, with the upstream pause allowance.
_SAMPLE_RATE = 24_000
_PEAK = 0.99
_CLAUSE_BOUNDARY = re.compile(r'(?<=[,;:—])\s+')


class _Phones(Protocol):
  @property
  def ids(self) -> tuple[int, ...]: ...

  @property
  def stress(self) -> tuple[int, ...]: ...


class _Session(Protocol):
  def run(
    self, output_names: list[str], input_feed: dict[str, np.ndarray]
  ) -> list[NDArray[np.float32]]: ...


class _SessionOptions(Protocol):
  intra_op_num_threads: int


class _Ort(Protocol):
  SessionOptions: Callable[[], _SessionOptions]
  InferenceSession: Callable[..., _Session]


class TinyVocaNativeEngine(LazyNativeEngine):
  """Run Tiny Voca's ONNX model and Dizione phonemizer on the CPU."""

  applies_speed = True

  def __init__(self, config: GatewayConfig) -> None:
    super().__init__(
      'tiny-voca',
      enabled=config.tiny_voca_enabled,
      models_dir=config.models_dir,
      device_mode='cpu',
    )
    self._phonemize: Callable[[str], _Phones | None] | None = None
    self._session: _Session | None = None

  def required_module_name(self) -> str | None:
    return 'onnxruntime'

  def install_hint(self) -> str | None:
    return 'uv sync --extra tiny-voca'

  def health_status(self) -> dict:
    return {**super().health_status(), 'model': _REPO}

  def _load_model(self) -> None:
    ort = cast(_Ort, import_module('onnxruntime'))
    phonemizer_factory = _import_phonemizer()
    download = import_module('huggingface_hub').hf_hub_download
    cache_dir = Path(self.models_dir).expanduser() / 'huggingface' / 'hub'

    def fetch(filename: str) -> Path:
      return Path(download(_REPO, filename, revision=_REVISION, cache_dir=cache_dir))

    model = fetch('onnx/model.onnx')
    for filename in _PHONEMIZER_FILES:
      fetch(f'phonemizer/{filename}')
    phonemize = phonemizer_factory(model.parent.parent / 'phonemizer')

    options = ort.SessionOptions()
    # The gateway already synthesizes multiple chunks concurrently.
    options.intra_op_num_threads = 2
    self._session = ort.InferenceSession(
      str(model), options, providers=['CPUExecutionProvider']
    )
    self._phonemize = phonemize
    self._device = 'cpu'

  def _run_inference(
    self, text: str, voice: str | None = None, speed: float = 1.0
  ) -> AudioChunk:
    session, phonemize = self._session, self._phonemize
    if session is None or phonemize is None:
      raise EngineError('tiny-voca model is not loaded')
    voice_id = _VOICES.get((voice or 'F2').upper())
    if voice_id is None:
      raise EngineError(f'tiny-voca voice must be F2 or M2, received: {voice!r}')

    utterances = (
      phones
      for sentence in SENTENCE_BOUNDARY.split(text.strip())
      for phones in _fit_utterance(sentence, phonemize)
    )
    waves = [_synthesize(session, phones, voice_id, speed) for phones in utterances]
    if not waves:
      raise EngineError('tiny-voca found no pronounceable English text')
    audio = np.concatenate(waves)
    peak = float(np.abs(audio).max(initial=0))
    if peak > _PEAK:
      audio *= _PEAK / peak
    return AudioChunk.from_float32(
      audio, sample_rate=_SAMPLE_RATE, speed_applied=speed if speed != 1.0 else None
    )


def _import_phonemizer() -> Callable[[Path], Callable[[str], _Phones | None]]:
  try:
    dizione = import_module('dizione')
    engine = import_module('dizione.engine')
  except ModuleNotFoundError as exc:
    if exc.name != 'dizione':
      raise
    raise EngineError(
      f'missing Dizione; install it with: uv pip install {_DIZIONE_SOURCE}'
    ) from exc

  def load(directory: Path) -> Callable[[str], _Phones | None]:
    phonemizer = engine.Phonemizer(directory)

    def phonemize(text: str) -> _Phones | None:
      pieces = phonemizer.pieces(text)
      if not any(word for mark, words in pieces if not mark for word in words):
        return None
      return cast(_Phones, dizione.tokenize(pieces))

    return phonemize

  return load


def _fit_utterance(
  text: str, phonemize: Callable[[str], _Phones | None]
) -> Iterator[_Phones]:
  # Terminal punctuation gives the model a natural ending on each clip.
  if not text.rstrip('"\'”\u2019)]»').endswith(
    ('.', '!', '?', '…', ',', ';', ':', '—')
  ):
    text += '.'
  phones = phonemize(text)
  if phones is None:
    return
  if len(phones.ids) <= _MAX_PHONEMES:
    yield phones
    return
  parts = _CLAUSE_BOUNDARY.split(text)
  if len(parts) == 1:
    words = text.split()
    if len(words) < 2:
      raise EngineError('tiny-voca found a word that exceeds its phoneme limit')
    middle = len(words) // 2
    parts = [' '.join(words[:middle]), ' '.join(words[middle:])]
  for part in parts:
    yield from _fit_utterance(part, phonemize)


def _synthesize(
  session: _Session, phones: _Phones, voice_id: int, speed: float
) -> NDArray[np.float32]:
  (audio,) = session.run(
    ['audio'],
    {
      'tokens': np.array([phones.ids], dtype=np.int64),
      'stress': np.array([phones.stress], dtype=np.int64),
      'voice': np.array([voice_id], dtype=np.int64),
      'speed': np.array([speed], dtype=np.float32),
      'pitch_shift': np.array([0.0], dtype=np.float32),
      'pitch_range': np.array([1.0], dtype=np.float32),
    },
  )
  if not audio.size or not np.isfinite(audio).all():
    raise EngineError('tiny-voca produced invalid audio')
  return audio[0]
