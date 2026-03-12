from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from tts_gateway.audio import align_chunk_format, encode_output, merge_chunks
from tts_gateway.chunking import chunk_text
from tts_gateway.config import EngineName, GatewayConfig
from tts_gateway.engines.base import AudioChunk, EngineError, TtsEngine
from tts_gateway.engines.kokoro_native import KokoroNativeEngine
from tts_gateway.engines.native_engine import LazyNativeEngine
from tts_gateway.engines.pocket_native import PocketNativeEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineAttempt:
  chunk_index: int
  engine: str
  ok: bool
  duration_ms: int
  error: str | None = None


@dataclass(frozen=True)
class SynthesisResult:
  payload: bytes
  content_type: str
  chunks_total: int
  attempts: list[EngineAttempt]


class SynthesisError(RuntimeError):
  def __init__(
    self,
    message: str,
    attempts: list[EngineAttempt],
    *,
    unavailable: bool = False,
    timed_out: bool = False,
  ) -> None:
    super().__init__(message)
    self.attempts = attempts
    self.unavailable = unavailable
    self.timed_out = timed_out


def _elapsed_ms(started: float) -> int:
  return int((time.perf_counter() - started) * 1000)


def _record_attempt(
  attempts: list[EngineAttempt],
  *,
  chunk_index: int,
  engine: str,
  ok: bool,
  started: float | None = None,
  error: str | None = None,
) -> None:
  attempts.append(
    EngineAttempt(
      chunk_index=chunk_index,
      engine=engine,
      ok=ok,
      duration_ms=0 if started is None else _elapsed_ms(started),
      error=error,
    )
  )


def _chunk_failure_message(chunk_index: int, last_error: Exception | None) -> str:
  message = f'chunk {chunk_index} failed across engine chain'
  if isinstance(last_error, EngineError):
    return f'{message}: {last_error}'
  if last_error is not None:
    return f'{message}: {type(last_error).__name__}: {last_error}'
  return message


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

  def engine_chain(self) -> list[str]:
    return self._engine_chain

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
    """Eagerly load all enabled native engines. Returns per-engine status."""
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

  async def _synthesize_chunk_with_chain(
    self,
    text: str,
    chunk_index: int,
    attempts: list[EngineAttempt],
    *,
    voice: str | None = None,
  ) -> AudioChunk:
    last_error: Exception | None = None
    any_available = False
    for engine_name in self._engine_chain:
      engine = self.engines.get(engine_name)

      if engine is None:
        logger.info(
          'engine-disabled-skipped',
          extra={'engine': engine_name, 'chunk_index': chunk_index},
        )
        _record_attempt(
          attempts,
          chunk_index=chunk_index,
          engine=engine_name,
          ok=False,
          error=f'{engine_name} is unavailable (disabled)',
        )
        continue

      any_available = True
      started = time.perf_counter()
      try:
        chunk = await asyncio.wait_for(
          engine.synthesize(text, voice=voice),
          timeout=self.config.engine_timeout_seconds,
        )
        _record_attempt(
          attempts,
          chunk_index=chunk_index,
          engine=engine_name,
          ok=True,
          started=started,
        )
        return chunk
      except TimeoutError:
        error_msg = (
          f'{engine_name} engine timed out after {self.config.engine_timeout_seconds}s'
        )
        last_error = EngineError(error_msg)
        _record_attempt(
          attempts,
          chunk_index=chunk_index,
          engine=engine_name,
          ok=False,
          started=started,
          error=error_msg,
        )
      except Exception as exc:
        last_error = exc
        _record_attempt(
          attempts,
          chunk_index=chunk_index,
          engine=engine_name,
          ok=False,
          started=started,
          error=str(exc),
        )

    if not any_available:
      raise SynthesisError(
        'all engines in chain are unavailable',
        attempts=attempts,
        unavailable=True,
      )

    raise SynthesisError(
      _chunk_failure_message(chunk_index, last_error), attempts=attempts
    )

  def _align_chunk(
    self,
    chunk: AudioChunk,
    reference_chunk: AudioChunk | None,
    chunk_index: int,
    attempts: list[EngineAttempt],
  ) -> AudioChunk:
    if reference_chunk is None:
      return chunk

    try:
      return align_chunk_format(
        chunk,
        reference_chunk,
        ffmpeg_path=self.config.ffmpeg_path,
      )
    except Exception as exc:
      raise SynthesisError(
        f'failed to align chunk {chunk_index} audio format: {exc}',
        attempts=attempts,
      ) from exc

  async def _synthesize_audio_chunks(
    self,
    chunks: list[str],
    attempts: list[EngineAttempt],
    *,
    voice: str | None,
  ) -> list[AudioChunk]:
    audio_chunks: list[AudioChunk] = []
    reference_chunk: AudioChunk | None = None

    for index, chunk in enumerate(chunks):
      audio_chunk = await self._synthesize_chunk_with_chain(
        chunk,
        index,
        attempts,
        voice=voice,
      )
      audio_chunks.append(
        self._align_chunk(audio_chunk, reference_chunk, index, attempts)
      )
      if reference_chunk is None:
        reference_chunk = audio_chunk

    return audio_chunks

  def _build_result(
    self,
    audio_chunks: list[AudioChunk],
    *,
    chunks_total: int,
    attempts: list[EngineAttempt],
  ) -> SynthesisResult:
    try:
      merged = merge_chunks(audio_chunks)
      payload, content_type = encode_output(
        merged,
        output_format=self.config.output_format,
        ffmpeg_path=self.config.ffmpeg_path,
      )
    except Exception as exc:
      raise SynthesisError(
        f'final audio assembly failed: {exc}', attempts=attempts
      ) from exc

    return SynthesisResult(
      payload=payload,
      content_type=content_type,
      chunks_total=chunks_total,
      attempts=attempts,
    )

  async def synthesize(
    self,
    text: str,
    *,
    voice: str | None = None,
    attempts: list[EngineAttempt] | None = None,
  ) -> SynthesisResult:
    chunks = chunk_text(text, self.config.chunk_max_chars)
    if not chunks:
      raise SynthesisError('text is empty after normalization', attempts=[])

    resolved_voice = voice or self.config.default_voice
    if attempts is None:
      attempts = []
    audio_chunks = await self._synthesize_audio_chunks(
      chunks,
      attempts,
      voice=resolved_voice,
    )
    return self._build_result(
      audio_chunks,
      chunks_total=len(chunks),
      attempts=attempts,
    )

  async def synthesize_with_timeout(
    self, text: str, *, voice: str | None = None
  ) -> SynthesisResult:
    attempts: list[EngineAttempt] = []
    try:
      return await asyncio.wait_for(
        self.synthesize(text, voice=voice, attempts=attempts),
        timeout=self.config.request_timeout_seconds,
      )
    except TimeoutError as exc:
      raise SynthesisError(
        'gateway request timed out',
        attempts=attempts,
        timed_out=True,
      ) from exc


def _resolve_engine(
  *,
  name: EngineName,
  enabled: bool,
  create_native: Callable[[], TtsEngine],
) -> TtsEngine | None:
  """Pick native engine or None based on configuration."""
  if enabled:
    logger.info('engine-resolved', extra={'engine': name, 'mode': 'native'})
    return create_native()
  logger.info('engine-resolved', extra={'engine': name, 'mode': 'disabled'})
  return None
