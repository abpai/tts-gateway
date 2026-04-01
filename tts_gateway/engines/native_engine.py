from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod

from tts_gateway.config import DeviceMode
from tts_gateway.engines.base import AudioChunk, EngineError, TtsEngine

logger = logging.getLogger(__name__)


class LazyNativeEngine(TtsEngine):
  """Base class for native in-process TTS engines with lazy model loading.

  Subclasses implement ``_load_model`` (one-time setup) and
  ``_run_inference`` (per-request synthesis).  Both are executed in a
  thread-pool executor so they never block the async event loop.
  """

  def __init__(
    self,
    name: str,
    *,
    enabled: bool,
    models_dir: str,
    device_mode: DeviceMode,
  ) -> None:
    self.name = name
    self.enabled = enabled
    self.models_dir = models_dir
    self.device_mode = device_mode

    self._loaded: bool = False
    self._load_error: str | None = None
    self._device: str = 'unknown'
    self._lock = asyncio.Lock()

  # ------------------------------------------------------------------
  # Subclass contract
  # ------------------------------------------------------------------

  @abstractmethod
  def _load_model(self) -> None:
    """Load model weights into memory.  Called once, under lock."""

  @abstractmethod
  def _run_inference(self, text: str, voice: str | None = None) -> AudioChunk:
    """Run TTS inference on *text*.  Called after model is loaded."""

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    if not self.enabled:
      raise EngineError(f'{self.name} engine is disabled')
    await self._ensure_loaded()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self._run_inference, text, voice)

  async def ensure_loaded(self) -> None:
    """Public wrapper so /warmup can call this directly."""
    await self._ensure_loaded()

  def health_status(self) -> dict:
    return {
      'mode': 'native',
      'enabled': self.enabled,
      'loaded': self._loaded,
      'device': self._device,
      'loadError': self._load_error,
    }

  def required_module_name(self) -> str | None:
    return None

  def install_hint(self) -> str | None:
    return None

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  async def _ensure_loaded(self) -> None:
    if self._loaded:
      return
    async with self._lock:
      if self._loaded:
        return
      logger.info('Loading %s engine (device=%s)...', self.name, self.device_mode)
      started = time.perf_counter()
      try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)
        self._loaded = True
        self._load_error = None
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
          'Loaded %s engine on %s in %dms',
          self.name,
          self._device,
          duration_ms,
        )
      except Exception as exc:
        load_error = self._format_load_error(exc)
        self._load_error = load_error
        logger.error('Failed to load %s engine: %s', self.name, load_error)
        raise EngineError(f'{self.name} failed to load: {load_error}') from exc

  def _format_load_error(self, exc: Exception) -> str:
    if self._is_missing_required_module(exc):
      hint = self.install_hint()
      if hint:
        return (
          f'missing dependency "{self.required_module_name()}"; install it with: {hint}'
        )
    return str(exc)

  def _is_missing_required_module(self, exc: Exception) -> bool:
    if not isinstance(exc, ModuleNotFoundError):
      return False
    required_module = self.required_module_name()
    if required_module is None:
      return False
    if exc.name == required_module:
      return True
    return f"No module named '{required_module}'" in str(exc)
