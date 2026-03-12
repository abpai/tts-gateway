from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib import import_module
from typing import Any, Protocol, cast

from tts_gateway.config import DeviceMode, GatewayConfig
from tts_gateway.engines.base import AudioChunk, EngineError
from tts_gateway.engines.native_engine import LazyNativeEngine

SAMPLE_RATE = 24_000
KOKORO_REPO_ID = 'hexgrad/Kokoro-82M'
logger = logging.getLogger(__name__)


class _KokoroPipeline(Protocol):
  def __call__(
    self,
    text: str,
    *,
    voice: str,
    speed: float,
  ) -> Iterable[tuple[Any, Any, Any]]: ...


class _KokoroModule(Protocol):
  def KPipeline(  # noqa: N802
    self,
    *,
    lang_code: str,
    device: str,
    repo_id: str,
  ) -> _KokoroPipeline: ...


class _TorchMpsBackend(Protocol):
  def is_available(self) -> bool: ...


class _TorchBackends(Protocol):
  mps: _TorchMpsBackend


class _TorchCuda(Protocol):
  def is_available(self) -> bool: ...


class _TorchModule(Protocol):
  cuda: _TorchCuda
  backends: _TorchBackends


class KokoroNativeEngine(LazyNativeEngine):
  """Native in-process Kokoro TTS engine using the ``kokoro`` PyTorch package."""

  def __init__(self, config: GatewayConfig) -> None:
    super().__init__(
      'kokoro',
      enabled=config.kokoro_enabled,
      models_dir=config.models_dir,
      device_mode=config.device_mode,
    )
    self._pipeline: _KokoroPipeline | None = None

  # ------------------------------------------------------------------
  # LazyNativeEngine contract
  # ------------------------------------------------------------------

  def _load_model(self) -> None:
    import os

    # Side-effect: sets HF_HOME so Hugging Face Hub caches models under
    # the configured models_dir rather than the default ~/.cache/huggingface.
    os.environ.setdefault('HF_HOME', os.path.join(self.models_dir, 'huggingface'))

    device = self._resolve_device(self.device_mode)
    self._device = device

    kokoro = cast(_KokoroModule, import_module('kokoro'))
    self._pipeline = kokoro.KPipeline(
      lang_code='a',
      device=device,
      repo_id=KOKORO_REPO_ID,
    )

  def _run_inference(self, text: str, voice: str | None = None) -> AudioChunk:
    import numpy as np

    pipeline = self._pipeline
    if pipeline is None:
      raise EngineError('kokoro model is not loaded')

    voice = voice or 'af_heart'

    segments: list = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=1.0):
      segments.append(audio)

    if not segments:
      raise EngineError('kokoro produced no audio output')

    return AudioChunk.from_float32(np.concatenate(segments), sample_rate=SAMPLE_RATE)

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _resolve_device(device_mode: DeviceMode) -> str:
    if device_mode == 'cpu':
      return 'cpu'

    try:
      torch = cast(_TorchModule, import_module('torch'))

      if device_mode == 'cuda':
        if torch.cuda.is_available():
          return 'cuda'
        logger.warning('kokoro cuda requested but unavailable; falling back to cpu')
        return 'cpu'

      if device_mode == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
          return 'mps'
        logger.warning('kokoro mps requested but unavailable; falling back to cpu')
        return 'cpu'

      if torch.cuda.is_available():
        return 'cuda'
      if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    except ImportError:
      pass

    return 'cpu'
