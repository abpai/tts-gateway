from __future__ import annotations

import threading
from importlib import import_module
from typing import Any, Protocol, cast

from tts_gateway.config import GatewayConfig
from tts_gateway.engines.base import AudioChunk, EngineError
from tts_gateway.engines.native_engine import LazyNativeEngine


class _PocketTensor(Protocol):
  def numpy(self) -> Any: ...


class _PocketModel(Protocol):
  sample_rate: int

  def get_state_for_audio_prompt(self, voice: str) -> Any: ...

  def generate_audio(self, voice_state: Any, text: str) -> _PocketTensor: ...


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

  # ------------------------------------------------------------------
  # LazyNativeEngine contract
  # ------------------------------------------------------------------

  def _load_model(self) -> None:
    module = import_module('pocket_tts')
    self._device = 'cpu'
    self._model = cast(_PocketModel, module.TTSModel.load_model())
    self._sample_rate = self._model.sample_rate

  def _run_inference(self, text: str, voice: str | None = None) -> AudioChunk:
    model = self._model
    if model is None:
      raise EngineError('pocket model is not loaded')

    voice = voice or 'alba'
    with self._voice_lock:
      if voice not in self._voice_states:
        self._voice_states[voice] = model.get_state_for_audio_prompt(voice)
      voice_state = self._voice_states[voice]

    audio_tensor = model.generate_audio(voice_state, text)
    audio_f32 = audio_tensor.numpy()

    if audio_f32.size == 0:
      raise EngineError('pocket produced no audio output')

    return AudioChunk.from_float32(audio_f32, sample_rate=self._sample_rate)
