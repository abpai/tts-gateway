"""Async worker that processes queued TTS synthesis jobs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from tts_gateway.engines.base import TtsEngine
from tts_gateway.jobs.store import JobStore
from tts_gateway.synthesis import SynthesisRequest, synthesize_to_disk

logger = logging.getLogger(__name__)


async def process_job(
  store: JobStore,
  job_key: str,
  request_json: str,
  engines: list[TtsEngine],
  artifacts_dir: Path,
  *,
  concurrency: int = 4,
  engine_timeout: float = 360.0,
  ffmpeg_path: str = 'ffmpeg',
) -> None:
  """Process a single job: synthesize to disk and update store."""
  request = SynthesisRequest.from_json(request_json)

  def _on_progress(done: int, total: int) -> None:
    store.update_progress(job_key, chunks_done=done, chunks_total=total)

  try:
    artifact = await synthesize_to_disk(
      request,
      engines,
      artifacts_dir,
      concurrency=concurrency,
      engine_timeout=engine_timeout,
      ffmpeg_path=ffmpeg_path,
      on_progress=_on_progress,
    )
    store.mark_ready(
      job_key,
      artifact_path=str(artifact.output_path),
      content_type=artifact.content_type,
      chunks_total=artifact.chunks_total,
    )
    logger.info('Job %s completed', job_key[:16])
  except Exception as exc:
    store.mark_failed(job_key, str(exc))
    logger.error('Job %s failed: %s', job_key[:16], exc)


async def run_worker(
  store: JobStore,
  engines: list[TtsEngine],
  artifacts_dir: Path,
  *,
  poll_seconds: float = 1.0,
  concurrency: int = 4,
  engine_timeout: float = 360.0,
  ffmpeg_path: str = 'ffmpeg',
  once: bool = False,
) -> None:
  """Main worker loop. Polls for queued jobs and processes them.

  Args:
    once: If True, process one job and return (useful for testing/cron).
  """
  logger.info('Worker started, polling every %.1fs', poll_seconds)

  while True:
    job = store.claim_next()
    if job is not None:
      logger.info('Processing job %s', job.key[:16])
      await process_job(
        store,
        job.key,
        job.request_json,
        engines,
        artifacts_dir,
        concurrency=concurrency,
        engine_timeout=engine_timeout,
        ffmpeg_path=ffmpeg_path,
      )
      if once:
        return
    else:
      if once:
        return
      await asyncio.sleep(poll_seconds)
