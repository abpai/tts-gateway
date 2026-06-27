"""Framework-free synthesis rendering.

Four entry points, no classes:
  plan_chunks          -- split text into ordered chunks
  synthesize_chunks    -- ordered parallel synthesis -> audio chunk stream
  stream_audio         -- encode-as-you-go -> byte stream
  synthesize_to_disk   -- write chunks + final artifact to disk
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

from tts_gateway.audio import (
  align_chunk_format,
  chunk_to_wav_bytes,
  encode_output,
  merge_chunks,
  wav_bytes_to_chunk,
)
from tts_gateway.chunking import chunk_text, stream_chunk_text
from tts_gateway.engines.base import (
  AudioChunk,
  StreamingTtsEngine,
  TtsEngine,
  supports_streaming,
)
from tts_gateway.types import ArtifactRef, RenderPlan, SynthesisSpec

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (chunks_done, chunks_total)


def plan_chunks(spec: SynthesisSpec) -> RenderPlan:
  """Split spec text into ordered chunks. Pure, no I/O."""
  chunks = chunk_text(spec.text, spec.chunk_max_chars)
  if not chunks:
    raise ValueError('text is empty after normalization')
  return RenderPlan(
    request_hash=spec.content_hash,
    chunks=tuple(chunks),
    voice=spec.voice,
    output_format=spec.output_format,
  )


def plan_stream_chunks(
  spec: SynthesisSpec,
  *,
  first_chunk_max_chars: int,
  stream_chunk_max_chars: int,
) -> RenderPlan:
  """Split spec text for streaming with a smaller first chunk. Pure, no I/O."""
  chunks = stream_chunk_text(
    spec.text,
    first_chunk_max_chars,
    stream_chunk_max_chars,
  )
  if not chunks:
    raise ValueError('text is empty after normalization')
  return RenderPlan(
    request_hash=spec.content_hash,
    chunks=tuple(chunks),
    voice=spec.voice,
    output_format=spec.output_format,
  )


async def synthesize_chunks(
  plan: RenderPlan,
  engines: list[TtsEngine],
  *,
  concurrency: int = 4,
  engine_timeout: float = 360.0,
  ffmpeg_path: str = 'ffmpeg',
) -> AsyncGenerator[AudioChunk, None]:
  """Yield AudioChunks in chunk-index order with parallel execution.

  Creates all futures up front (gated by semaphore) and awaits them
  in index order. This gives parallel engine utilization with ordered
  emission — the foundation for streaming.
  """
  semaphore = asyncio.Semaphore(concurrency)

  async def _synth_one(index: int, text: str) -> AudioChunk:
    async with semaphore:
      logger.debug(
        'chunk-started', extra={'chunk_index': index, 'chunk_chars': len(text)}
      )
      started = time.perf_counter()
      chunk = await _try_engines(text, plan.voice, engines, timeout=engine_timeout)
      elapsed = int((time.perf_counter() - started) * 1000)
      logger.debug('chunk-done', extra={'chunk_index': index, 'elapsed_ms': elapsed})
      return chunk

  futures = [
    asyncio.ensure_future(_synth_one(i, text)) for i, text in enumerate(plan.chunks)
  ]

  try:
    reference: AudioChunk | None = None
    for i in range(len(plan.chunks)):
      chunk = await futures[i]
      if reference is not None:
        chunk = align_chunk_format(chunk, reference, ffmpeg_path)
      else:
        reference = chunk
      yield chunk
  finally:
    for f in futures:
      if not f.done():
        f.cancel()
    await asyncio.gather(*futures, return_exceptions=True)


async def stream_pcm(
  spec: SynthesisSpec,
  engines: list[TtsEngine],
  *,
  concurrency: int = 4,
  engine_timeout: float = 360.0,
  ffmpeg_path: str = 'ffmpeg',
  stream_first_chunk_max_chars: int = 180,
  stream_chunk_max_chars: int | None = None,
) -> tuple[AudioChunk, AsyncGenerator[AudioChunk, None]]:
  """Return first-chunk metadata and remaining ordered audio chunks."""
  chunks = _stream_audio_chunks(
    spec,
    engines,
    concurrency=concurrency,
    engine_timeout=engine_timeout,
    ffmpeg_path=ffmpeg_path,
    stream_first_chunk_max_chars=stream_first_chunk_max_chars,
    stream_chunk_max_chars=stream_chunk_max_chars,
  )
  try:
    first = await chunks.__anext__()
  except Exception:
    await chunks.aclose()
    raise
  return first, chunks


async def stream_audio(
  spec: SynthesisSpec,
  engines: list[TtsEngine],
  *,
  concurrency: int = 4,
  engine_timeout: float = 360.0,
  ffmpeg_path: str = 'ffmpeg',
  stream_first_chunk_max_chars: int = 180,
  stream_chunk_max_chars: int | None = None,
) -> AsyncGenerator[bytes, None]:
  """Yield encoded audio bytes chunk-by-chunk.

  For MP3: each chunk is independently encoded and yielded.
  For WAV: yields raw PCM bytes (caller handles framing).
  """
  async for audio_chunk in _stream_audio_chunks(
    spec,
    engines,
    concurrency=concurrency,
    engine_timeout=engine_timeout,
    ffmpeg_path=ffmpeg_path,
    stream_first_chunk_max_chars=stream_first_chunk_max_chars,
    stream_chunk_max_chars=stream_chunk_max_chars,
  ):
    if spec.output_format == 'mp3':
      payload, _ = encode_output(audio_chunk, 'mp3', ffmpeg_path)
      yield payload
    else:
      yield audio_chunk.pcm_bytes


async def _stream_audio_chunks(
  spec: SynthesisSpec,
  engines: list[TtsEngine],
  *,
  concurrency: int,
  engine_timeout: float,
  ffmpeg_path: str,
  stream_first_chunk_max_chars: int,
  stream_chunk_max_chars: int | None,
) -> AsyncGenerator[AudioChunk, None]:
  opened = await _open_native_stream(
    spec.text, spec.voice, engines, timeout=engine_timeout
  )
  if opened is not None:
    first, stream = opened
    logger.info('Streaming native PCM from engine')
    async for chunk in _native_pcm_chunks(
      first,
      stream,
      ffmpeg_path=ffmpeg_path,
      timeout=engine_timeout,
      include_first=True,
    ):
      yield chunk
    return

  plan = _stream_chunk_plan(spec, stream_first_chunk_max_chars, stream_chunk_max_chars)
  logger.info('Streaming %d chunk(s)', len(plan.chunks))
  async for chunk in synthesize_chunks(
    plan,
    engines,
    concurrency=concurrency,
    engine_timeout=engine_timeout,
    ffmpeg_path=ffmpeg_path,
  ):
    yield chunk


def _stream_chunk_plan(
  spec: SynthesisSpec,
  first_chunk_max_chars: int,
  stream_chunk_max_chars: int | None,
) -> RenderPlan:
  chunk_limit = (
    stream_chunk_max_chars
    if stream_chunk_max_chars is not None
    else spec.chunk_max_chars
  )
  return plan_stream_chunks(
    spec,
    first_chunk_max_chars=first_chunk_max_chars,
    stream_chunk_max_chars=chunk_limit,
  )


async def synthesize_to_disk(
  spec: SynthesisSpec,
  engines: list[TtsEngine],
  output_dir: Path,
  *,
  concurrency: int = 4,
  engine_timeout: float = 360.0,
  ffmpeg_path: str = 'ffmpeg',
  on_progress: ProgressCallback | None = None,
) -> ArtifactRef:
  """Synthesize all chunks to disk, produce a final merged output file.

  Layout:
    {output_dir}/{content_hash}/
      chunk_000.wav
      chunk_001.wav
      output.mp3  (or output.wav)

  Cache hit: returns immediately if output file exists.
  Resume: skips chunks whose WAV files already exist on disk.
  Filesystem lock: uses fcntl.flock to serialize concurrent writers.
  """
  plan = plan_chunks(spec)
  job_dir = output_dir / plan.request_hash
  job_dir.mkdir(parents=True, exist_ok=True)

  ext = 'mp3' if spec.output_format == 'mp3' else 'wav'
  final_path = job_dir / f'output.{ext}'
  content_type = 'audio/mpeg' if ext == 'mp3' else 'audio/wav'

  # Cache hit — no lock needed
  if final_path.exists():
    return ArtifactRef(
      request_hash=plan.request_hash,
      output_path=final_path,
      content_type=content_type,
      chunks_total=len(plan.chunks),
      duration_ms=0,
    )

  # Acquire exclusive lock to serialize concurrent writers
  lock_path = job_dir / 'lock'
  lock_file = open(lock_path, 'w')
  try:
    fcntl.flock(lock_file, fcntl.LOCK_EX)

    # Re-check cache after acquiring lock (another writer may have finished)
    if final_path.exists():
      return ArtifactRef(
        request_hash=plan.request_hash,
        output_path=final_path,
        content_type=content_type,
        chunks_total=len(plan.chunks),
        duration_ms=0,
      )

    logger.info('Synthesizing %d chunk(s) to disk at %s', len(plan.chunks), job_dir)

    # Check which chunks already exist (resume support)
    existing = {
      i for i in range(len(plan.chunks)) if (job_dir / f'chunk_{i:03d}.wav').exists()
    }
    if existing:
      logger.info(
        'Resuming: %d/%d chunks already on disk', len(existing), len(plan.chunks)
      )

    started = time.perf_counter()
    audio_chunks: dict[int, AudioChunk] = {}

    if on_progress:
      on_progress(len(existing), len(plan.chunks))

    # Load existing chunks from disk (no re-synthesis)
    for i in existing:
      chunk_path = job_dir / f'chunk_{i:03d}.wav'
      audio_chunks[i] = wav_bytes_to_chunk(chunk_path.read_bytes())

    # Build a partial plan with only missing chunks
    missing_indices = [i for i in range(len(plan.chunks)) if i not in existing]
    if missing_indices:
      missing_plan = RenderPlan(
        request_hash=plan.request_hash,
        chunks=tuple(plan.chunks[i] for i in missing_indices),
        voice=plan.voice,
        output_format=plan.output_format,
      )
      synth_iter = synthesize_chunks(
        missing_plan,
        engines,
        concurrency=concurrency,
        engine_timeout=engine_timeout,
        ffmpeg_path=ffmpeg_path,
      )
      j = 0
      async for audio_chunk in synth_iter:
        orig_index = missing_indices[j]
        chunk_path = job_dir / f'chunk_{orig_index:03d}.wav'
        chunk_path.write_bytes(chunk_to_wav_bytes(audio_chunk))
        audio_chunks[orig_index] = audio_chunk
        j += 1
        if on_progress:
          on_progress(len(existing) + j, len(plan.chunks))

    # Assemble in original order
    ordered_chunks = [audio_chunks[i] for i in range(len(plan.chunks))]

    # Merge and encode
    merged = merge_chunks(ordered_chunks)
    payload, _ = encode_output(merged, spec.output_format, ffmpeg_path)
    final_path.write_bytes(payload)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info('Synthesis complete in %dms, wrote %s', elapsed_ms, final_path)

    return ArtifactRef(
      request_hash=plan.request_hash,
      output_path=final_path,
      content_type=content_type,
      chunks_total=len(plan.chunks),
      duration_ms=elapsed_ms,
    )
  finally:
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()


async def _try_engines(
  text: str,
  voice: str,
  engines: list[TtsEngine],
  *,
  timeout: float,
) -> AudioChunk:
  """Try each engine in order until one succeeds."""
  if not engines:
    raise RuntimeError('no engines available')

  last_error: Exception | None = None
  for engine in engines:
    try:
      return await asyncio.wait_for(
        engine.synthesize(text, voice=voice),
        timeout=timeout,
      )
    except Exception as exc:
      last_error = exc
      logger.warning(
        'engine-fallback',
        extra={'engine': getattr(engine, 'name', '?'), 'error': str(exc)},
      )

  raise RuntimeError(f'all engines failed: {last_error}') from last_error


async def _open_native_stream(
  text: str,
  voice: str,
  engines: list[TtsEngine],
  *,
  timeout: float,
) -> tuple[AudioChunk, AsyncGenerator[AudioChunk, None]] | None:
  """Open the first native stream that yields a chunk before timeout."""
  candidates = _streaming_candidates(engines)
  if not candidates:
    return None

  last_error: Exception | None = None
  for engine in candidates:
    stream = engine.stream_synthesize(text, voice=voice)
    try:
      first = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
      return first, stream
    except TimeoutError as exc:
      last_error = exc
      await stream.aclose()
      logger.warning(
        'stream-engine-fallback',
        extra={'engine': getattr(engine, 'name', '?'), 'error': 'timeout'},
      )
    except StopAsyncIteration:
      await stream.aclose()
    except Exception as exc:
      last_error = exc
      await stream.aclose()
      logger.warning(
        'stream-engine-fallback',
        extra={'engine': getattr(engine, 'name', '?'), 'error': str(exc)},
      )

  if last_error is not None and len(candidates) == len(engines):
    raise last_error
  return None


def _streaming_candidates(engines: list[TtsEngine]) -> list[StreamingTtsEngine]:
  if not engines or not supports_streaming(engines[0]):
    return []

  candidates: list[StreamingTtsEngine] = []
  for engine in engines:
    if supports_streaming(engine):
      candidates.append(engine)
  return candidates


async def _aligned_native_tail(
  reference: AudioChunk,
  stream: AsyncGenerator[AudioChunk, None],
  *,
  ffmpeg_path: str,
  timeout: float,
) -> AsyncGenerator[AudioChunk, None]:
  while True:
    try:
      chunk = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
    except StopAsyncIteration:
      return
    yield align_chunk_format(chunk, reference, ffmpeg_path)


async def _native_pcm_chunks(
  first: AudioChunk,
  stream: AsyncGenerator[AudioChunk, None],
  *,
  ffmpeg_path: str,
  timeout: float,
  include_first: bool,
) -> AsyncGenerator[AudioChunk, None]:
  """Yield native-stream chunks with format alignment."""
  try:
    if include_first:
      yield first
    async for chunk in _aligned_native_tail(
      first,
      stream,
      ffmpeg_path=ffmpeg_path,
      timeout=timeout,
    ):
      yield chunk
  finally:
    await stream.aclose()
