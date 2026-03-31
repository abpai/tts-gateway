"""TTS Gateway: engine lifecycle + thin adapter over synthesis core."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace

from tts_gateway.audio import encode_output, merge_chunks
from tts_gateway.config import EngineName, GatewayConfig, OutputFormat
from tts_gateway.engines.base import AudioChunk, TtsEngine
from tts_gateway.engines.kokoro_native import KokoroNativeEngine
from tts_gateway.engines.native_engine import LazyNativeEngine
from tts_gateway.engines.pocket_native import PocketNativeEngine
from tts_gateway.synthesis import (
  SynthesisRequest,
  plan_chunks,
  stream_synthesis,
  synthesize_chunks,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisResult:
  payload: bytes
  content_type: str
  chunks_total: int


class SynthesisError(RuntimeError):
  def __init__(
    self,
    message: str,
    *,
    unavailable: bool = False,
    timed_out: bool = False,
  ) -> None:
    super().__init__(message)
    self.unavailable = unavailable
    self.timed_out = timed_out


def _default_chunk_concurrency() -> int:
  return max(1, min(4, os.cpu_count() or 1))


class TtsGateway:
  def __init__(self, config: GatewayConfig) -> None:
    self.config = config
    self.engines: dict[str, TtsEngine | None] = {
      'kokoro': _resolve_engine(
        name='kokoro',
        enabled=config.kokoro_enabled,
        create_native=lambda: KokoroNativeEngine(config),
      ),
      'pocket': _resolve_engine(
        name='pocket',
        enabled=config.pocket_enabled,
        create_native=lambda: PocketNativeEngine(config),
      ),
    }
    chain: list[str] = [config.primary_engine]
    fallback = config.fallback_engine
    if fallback and fallback != config.primary_engine:
      chain.append(fallback)
    self._engine_chain = chain
    self._chunk_concurrency = _default_chunk_concurrency()

  def engine_chain(self) -> list[str]:
    return self._engine_chain

  def chunk_concurrency(self) -> int:
    return self._chunk_concurrency

  def engine_list(self) -> list[TtsEngine]:
    """Return the active engine fallback chain as a list."""
    engines: list[TtsEngine] = []
    for name in self._engine_chain:
      engine = self.engines.get(name)
      if engine is not None:
        engines.append(engine)
    return engines

  def engine_info(self) -> dict[str, dict]:
    """Per-engine status for /health reporting."""
    info: dict[str, dict] = {}
    for name in ('kokoro', 'pocket'):
      engine = self.engines.get(name)
      if engine is None:
        info[name] = {'mode': 'disabled'}
      elif isinstance(engine, LazyNativeEngine):
        info[name] = engine.health_status()
      else:
        info[name] = {'mode': 'unknown'}
    return info

  async def warmup(self) -> dict[str, dict]:
    """Eagerly load all enabled native engines."""
    results: dict[str, dict] = {}
    for name, engine in self.engines.items():
      if isinstance(engine, LazyNativeEngine) and engine.enabled:
        try:
          await engine.ensure_loaded()
          status = engine.health_status()
          results[name] = {'loaded': True, 'device': status['device']}
        except Exception as exc:
          results[name] = {'loaded': False, 'error': str(exc)}
    return results

  def _make_request(self, text: str, *, voice: str | None = None) -> SynthesisRequest:
    """Build a SynthesisRequest from gateway config + caller args."""
    return SynthesisRequest(
      text=text,
      voice=voice or self.config.default_voice or '',
      output_format=self.config.output_format,
      chunk_max_chars=self.config.chunk_max_chars,
      pipeline_version=self.config.pipeline_version,
    )

  async def synthesize(
    self,
    text: str,
    *,
    voice: str | None = None,
  ) -> SynthesisResult:
    """Synthesize text to audio bytes (buffered, backward-compatible)."""
    engines = self.engine_list()
    if not engines:
      raise SynthesisError(
        'all engines in chain are unavailable',
        unavailable=True,
      )

    request = self._make_request(text, voice=voice)
    plan = plan_chunks(request)

    logger.info(
      'Synthesizing %d chunk(s), %d chars total',
      len(plan.chunks),
      sum(len(c) for c in plan.chunks),
    )

    started = time.perf_counter()
    audio_chunks: list[AudioChunk] = []
    try:
      async for audio_chunk in synthesize_chunks(
        plan,
        engines,
        concurrency=self._chunk_concurrency,
        engine_timeout=self.config.engine_timeout_seconds,
        ffmpeg_path=self.config.ffmpeg_path,
      ):
        audio_chunks.append(audio_chunk)
    except RuntimeError as exc:
      raise SynthesisError(str(exc)) from exc

    merged = merge_chunks(audio_chunks)
    payload, content_type = encode_output(
      merged, self.config.output_format, self.config.ffmpeg_path
    )

    logger.info(
      'Synthesized %d chunk(s) in %.1fs',
      len(plan.chunks),
      time.perf_counter() - started,
    )

    return SynthesisResult(
      payload=payload,
      content_type=content_type,
      chunks_total=len(plan.chunks),
    )

  async def synthesize_with_timeout(
    self, text: str, *, voice: str | None = None
  ) -> SynthesisResult:
    try:
      return await asyncio.wait_for(
        self.synthesize(text, voice=voice),
        timeout=self.config.request_timeout_seconds,
      )
    except TimeoutError as exc:
      raise SynthesisError(
        'gateway request timed out',
        timed_out=True,
      ) from exc

  async def stream(
    self,
    text: str,
    *,
    voice: str | None = None,
    output_format: OutputFormat | None = None,
  ) -> AsyncIterator[bytes]:
    """Stream encoded audio bytes as chunks complete.

    Raises SynthesisError eagerly (before yielding) if engines unavailable.
    """
    engines = self.engine_list()
    if not engines:
      raise SynthesisError(
        'all engines in chain are unavailable',
        unavailable=True,
      )

    request = self._make_request(text, voice=voice)
    if output_format:
      request = replace(request, output_format=output_format)

    async for chunk_bytes in stream_synthesis(
      request,
      engines,
      concurrency=self._chunk_concurrency,
      engine_timeout=self.config.engine_timeout_seconds,
      ffmpeg_path=self.config.ffmpeg_path,
    ):
      yield chunk_bytes


def _resolve_engine(
  *,
  name: EngineName,
  enabled: bool,
  create_native: Callable[[], TtsEngine],
) -> TtsEngine | None:
  """Pick native engine or None based on configuration."""
  if enabled:
    logger.debug('engine-resolved', extra={'engine': name, 'mode': 'native'})
    return create_native()
  logger.debug('engine-resolved', extra={'engine': name, 'mode': 'disabled'})
  return None
