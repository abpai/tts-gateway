from __future__ import annotations

import logging
from collections.abc import Iterable, MutableMapping
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from tts_gateway.config import DeviceMode, GatewayConfig
from tts_gateway.engines.base import AudioChunk, EngineError
from tts_gateway.engines.native_engine import LazyNativeEngine

SAMPLE_RATE = 24_000
KOKORO_REPO_ID = 'hexgrad/Kokoro-82M'
KOKORO_MODEL_FILE = 'kokoro-v1_0.pth'
logger = logging.getLogger(__name__)


class _KokoroPipeline(Protocol):
  voices: MutableMapping[str, Any]

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
    model: Any = ...,
  ) -> _KokoroPipeline: ...


class _KModelFactory(Protocol):
  def __call__(
    self,
    *,
    repo_id: str,
    config: str,
    model: str,
  ) -> Any: ...


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

  def required_module_name(self) -> str | None:
    return 'kokoro'

  def install_hint(self) -> str | None:
    return 'uv sync --group dev --extra kokoro'

  # ------------------------------------------------------------------
  # LazyNativeEngine contract
  # ------------------------------------------------------------------

  def _load_model(self) -> None:
    import os

    hf_home = os.path.join(self.models_dir, 'huggingface')
    # Side-effect: sets HF_HOME so Hugging Face Hub caches models under
    # the configured models_dir rather than the default ~/.cache/huggingface.
    os.environ.setdefault('HF_HOME', hf_home)

    device = self._resolve_device(self.device_mode)
    self._device = device

    kokoro = cast(_KokoroModule, import_module('kokoro'))

    # Use cached model files directly if available, avoiding HF network calls.
    snapshot_dir = self._find_cached_snapshot(Path(hf_home))
    if snapshot_dir:
      logger.info('Using cached model from %s', snapshot_dir)
      config_path = str(snapshot_dir / 'config.json')
      model_path = str(snapshot_dir / KOKORO_MODEL_FILE)
      kokoro_model = import_module('kokoro.model')
      model_factory = cast(_KModelFactory, kokoro_model.KModel)

      model = model_factory(
        repo_id=KOKORO_REPO_ID,
        config=config_path,
        model=model_path,
      )
      model = model.to(device).eval()
      pipeline = kokoro.KPipeline(
        lang_code='a',
        device=device,
        repo_id=KOKORO_REPO_ID,
        model=model,
      )
      # Preload cached voice files so load_single_voice skips hf_hub_download
      voices_dir = snapshot_dir / 'voices'
      if voices_dir.is_dir():
        load_torch_file = cast(Any, import_module('torch')).load
        for voice_file in voices_dir.glob('*.pt'):
          voice_name = voice_file.stem
          pipeline.voices[voice_name] = load_torch_file(
            str(voice_file), weights_only=True
          )
      self._pipeline = pipeline
    else:
      self._pipeline = kokoro.KPipeline(
        lang_code='a',
        device=device,
        repo_id=KOKORO_REPO_ID,
      )

  @staticmethod
  def _find_cached_snapshot(hf_home: Path) -> Path | None:
    """Find the latest HF Hub snapshot dir for Kokoro, if it exists."""
    snapshots_dir = hf_home / 'hub' / 'models--hexgrad--Kokoro-82M' / 'snapshots'
    if not snapshots_dir.is_dir():
      return None
    # Pick the first snapshot that has both config.json and the model file
    for snapshot in sorted(snapshots_dir.iterdir()):
      if (snapshot / 'config.json').exists() and (
        snapshot / KOKORO_MODEL_FILE
      ).exists():
        return snapshot
    return None

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
