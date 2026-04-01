from __future__ import annotations

import logging
import threading
import time
from importlib import import_module
from typing import Any, Protocol, cast

from tts_gateway.config import GatewayConfig
from tts_gateway.engines.base import AudioChunk, EngineError
from tts_gateway.engines.native_engine import LazyNativeEngine

logger = logging.getLogger(__name__)


class _PocketTensor(Protocol):
  def numpy(self) -> Any: ...


class _PocketModel(Protocol):
  sample_rate: int

  def get_state_for_audio_prompt(self, voice: str) -> Any: ...

  def generate_audio(self, voice_state: Any, text: str) -> _PocketTensor: ...

  def generate_audio_stream(
    self,
    model_state: Any,
    text_to_generate: str,
  ) -> Any: ...


class PocketNativeEngine(LazyNativeEngine):
  """Native in-process Pocket TTS engine using the ``pocket-tts`` package.

  Pocket TTS is CPU-only by design — the ``device_mode`` config is
  ignored and always resolves to ``cpu``.
  """

  def __init__(self, config: GatewayConfig) -> None:
    super().__init__(
      'pocket',
      enabled=config.pocket_enabled,
      models_dir=config.models_dir,
      device_mode='cpu',
    )
    self._model: _PocketModel | None = None
    self._voice_states: dict[str, Any] = {}
    self._voice_lock = threading.Lock()
    self._sample_rate: int = 24_000

  def required_module_name(self) -> str | None:
    return 'pocket_tts'

  def install_hint(self) -> str | None:
    return 'uv sync --group dev --extra pocket'

  # ------------------------------------------------------------------
  # LazyNativeEngine contract
  # ------------------------------------------------------------------

  def _load_model(self) -> None:
    module = import_module('pocket_tts')
    self._device = 'cpu'
    self._model = cast(_PocketModel, module.TTSModel.load_model())
    self._sample_rate = self._model.sample_rate

  def _run_inference(self, text: str, voice: str | None = None) -> AudioChunk:
    import numpy as np

    model = self._model
    if model is None:
      raise EngineError('pocket model is not loaded')

    voice = voice or 'alba'
    with self._voice_lock:
      if voice not in self._voice_states:
        self._voice_states[voice] = model.get_state_for_audio_prompt(voice)
      voice_state = self._voice_states[voice]

    chunks: list[Any] = []
    total_samples = 0
    started = time.perf_counter()
    for index, audio_chunk in enumerate(
      model.generate_audio_stream(voice_state, text),
      start=1,
    ):
      chunks.append(audio_chunk)
      total_samples += int(audio_chunk.shape[0])
      generated_ms = int((total_samples / self._sample_rate) * 1000)
      logger.debug(
        'pocket-stream-progress',
        extra={
          'voice': voice,
          'stream_chunk_index': index,
          'generated_ms': generated_ms,
          'elapsed_ms': int((time.perf_counter() - started) * 1000),
        },
      )

    if not chunks:
      raise EngineError('pocket produced no audio output')

    audio_f32 = np.concatenate([chunk.numpy() for chunk in chunks])
    if audio_f32.size == 0:
      raise EngineError('pocket produced no audio output')

    return AudioChunk.from_float32(audio_f32, sample_rate=self._sample_rate)
