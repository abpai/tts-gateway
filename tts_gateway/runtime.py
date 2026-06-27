"""JobRuntime: the single application service for TTS synthesis.

Owns engine lifecycle, job submission, execution, and worker coordination.
Everything else (FastAPI routes, CLI commands) is a thin adapter over this.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

from tts_gateway.config import EngineName, GatewayConfig
from tts_gateway.engines.base import TtsEngine
from tts_gateway.engines.cosyvoice_sidecar import CosyVoiceSidecarEngine
from tts_gateway.engines.kokoro_native import KokoroNativeEngine
from tts_gateway.engines.native_engine import LazyNativeEngine
from tts_gateway.engines.pocket_native import PocketNativeEngine
from tts_gateway.jobs.store import JobRecord, JobStore
from tts_gateway.render import synthesize_to_disk
from tts_gateway.types import ArtifactRef, JobView, SynthesisSpec

logger = logging.getLogger(__name__)


class NoEnginesError(RuntimeError):
  """All engines in the chain are disabled or unavailable."""


def _default_concurrency() -> int:
  return max(1, min(4, os.cpu_count() or 1))


def _resolve_engine(
  *,
  name: EngineName,
  enabled: bool,
  create_native: Callable[[], TtsEngine],
) -> TtsEngine | None:
  if enabled:
    logger.debug('engine-resolved', extra={'engine': name, 'mode': 'native'})
    return create_native()
  logger.debug('engine-resolved', extra={'engine': name, 'mode': 'disabled'})
  return None


def _record_to_view(record: JobRecord) -> JobView:
  return JobView(
    key=record.key,
    status=record.status,
    created_at=record.created_at,
    started_at=record.started_at,
    completed_at=record.completed_at,
    chunks_total=record.chunks_total,
    chunks_done=record.chunks_done,
    content_type=record.content_type,
    error=record.error,
  )


def _cache_namespace(config: GatewayConfig, engine_chain: list[str]) -> str:
  parts = [f'engines={",".join(engine_chain)}']
  if 'cosyvoice' in engine_chain:
    parts.append(f'cosyvoice={config.cosyvoice_base_url}')
  return '|'.join(parts)


class JobRuntime:
  """The single application service for TTS synthesis.

  Owns engine registry, job store, artifact storage, and worker coordination.
  """

  def __init__(self, config: GatewayConfig) -> None:
    self.config = config

    # Engine registry
    self._engine_map: dict[str, TtsEngine | None] = {
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
      'cosyvoice': _resolve_engine(
        name='cosyvoice',
        enabled=config.cosyvoice_enabled,
        create_native=lambda: CosyVoiceSidecarEngine(config),
      ),
    }
    chain: list[str] = [config.primary_engine]
    if config.fallback_engine and config.fallback_engine != config.primary_engine:
      chain.append(config.fallback_engine)
    self._engine_chain = chain
    self._concurrency = _default_concurrency()

    # Job infrastructure
    data_dir = Path(config.data_dir)
    self._artifacts_dir = data_dir / 'artifacts'
    self._artifacts_dir.mkdir(parents=True, exist_ok=True)
    self._store = JobStore(data_dir / 'jobs.db')

  # --- Spec construction (one place) ---

  def make_spec(self, text: str, voice: str | None = None) -> SynthesisSpec:
    return SynthesisSpec(
      text=text,
      voice=voice or self.config.default_voice or '',
      output_format=self.config.output_format,
      chunk_max_chars=self.config.chunk_max_chars,
      pipeline_version=self.config.pipeline_version,
      cache_namespace=_cache_namespace(self.config, self._engine_chain),
    )

  # --- Job API ---

  def submit(self, spec: SynthesisSpec) -> JobView:
    """Create or return an existing job. Idempotent on content_hash."""
    record = self._store.create_or_get(spec.content_hash, spec.to_json())
    return _record_to_view(record)

  def get(self, key: str) -> JobView | None:
    record = self._store.get(key)
    if record is None:
      return None
    return _record_to_view(record)

  def get_artifact_path(self, key: str) -> tuple[Path, str] | None:
    """Returns (path, content_type) if the job is ready, else None."""
    record = self._store.get(key)
    if record is None or record.status != 'ready' or not record.artifact_path:
      return None
    return Path(record.artifact_path), record.content_type or 'application/octet-stream'

  # --- Execution ---

  async def run_until_complete(
    self,
    spec: SynthesisSpec,
    *,
    timeout: float | None = None,
  ) -> ArtifactRef:
    """Sync adapter: submit job, execute inline, return artifact.

    Ownership algorithm:
    1. If already ready → return cached artifact.
    2. Try to claim(key) → if won, execute inline.
    3. If someone else owns it → poll until complete.
    """
    key = spec.content_hash
    job = self._store.create_or_get(key, spec.to_json())

    # Already done
    if job.status == 'ready' and job.artifact_path:
      return ArtifactRef(
        request_hash=key,
        output_path=Path(job.artifact_path),
        content_type=job.content_type or 'application/octet-stream',
        chunks_total=job.chunks_total or 0,
        duration_ms=0,
      )

    # Try to claim
    if self._store.claim(key):
      coro = self._execute_job(key, spec)
      if timeout:
        return await asyncio.wait_for(coro, timeout=timeout)
      return await coro

    # Someone else owns it — poll
    return await self._poll_until_complete(key, timeout=timeout)

  async def _execute_job(self, key: str, spec: SynthesisSpec) -> ArtifactRef:
    """Execute a claimed job inline."""
    engines = self.engines
    if not engines:
      self._store.mark_failed(key, 'no engines available')
      raise NoEnginesError('no engines available')

    def _on_progress(done: int, total: int) -> None:
      self._store.update_progress(key, chunks_done=done, chunks_total=total)

    try:
      artifact = await synthesize_to_disk(
        spec,
        engines,
        self._artifacts_dir,
        concurrency=self._concurrency,
        engine_timeout=self.config.engine_timeout_seconds,
        ffmpeg_path=self.config.ffmpeg_path,
        on_progress=_on_progress,
      )
      self._store.mark_encoding(key)
      self._store.mark_ready(
        key,
        artifact_path=str(artifact.output_path),
        content_type=artifact.content_type,
        chunks_total=artifact.chunks_total,
      )
      return artifact
    except asyncio.CancelledError:
      # Timeout — requeue so worker or next sync request can resume.
      # Partial chunks on disk are resumable via flock.
      self._store.requeue(key)
      raise
    except Exception as exc:
      self._store.mark_failed(key, str(exc))
      raise

  async def _poll_until_complete(
    self, key: str, *, timeout: float | None = None
  ) -> ArtifactRef:
    """Poll the store until the job is ready or failed."""
    deadline = None
    if timeout:
      deadline = asyncio.get_event_loop().time() + timeout

    while True:
      record = self._store.get(key)
      if record and record.status == 'ready' and record.artifact_path:
        return ArtifactRef(
          request_hash=key,
          output_path=Path(record.artifact_path),
          content_type=record.content_type or 'application/octet-stream',
          chunks_total=record.chunks_total or 0,
          duration_ms=0,
        )
      if record and record.status == 'failed':
        raise RuntimeError(f'job failed: {record.error}')

      if deadline and asyncio.get_event_loop().time() >= deadline:
        raise TimeoutError('timed out waiting for job to complete')

      await asyncio.sleep(0.5)

  async def run_next(self) -> bool:
    """Claim and process the oldest queued job. Returns False if queue empty."""
    job = self._store.claim_next()
    if job is None:
      return False

    logger.info('Processing job %s', job.key[:16])
    spec = SynthesisSpec.from_json(job.request_json)
    try:
      await self._execute_job(job.key, spec)
      logger.info('Job %s completed', job.key[:16])
    except Exception as exc:
      logger.error('Job %s failed: %s', job.key[:16], exc)
    return True

  # --- Engine lifecycle ---

  @property
  def engines(self) -> list[TtsEngine]:
    """Active engine fallback chain."""
    result: list[TtsEngine] = []
    for name in self._engine_chain:
      engine = self._engine_map.get(name)
      if engine is not None:
        result.append(engine)
    return result

  def engine_chain(self) -> list[str]:
    return self._engine_chain

  def engine_info(self) -> dict[str, dict]:
    """Per-engine status for /health reporting."""
    info: dict[str, dict] = {}
    for name in ('kokoro', 'pocket', 'cosyvoice'):
      engine = self._engine_map.get(name)
      if engine is None:
        info[name] = {'mode': 'disabled'}
      else:
        info[name] = engine.health_status()
    return info

  async def warmup(self) -> dict[str, dict]:
    """Eagerly load all enabled native engines."""
    results: dict[str, dict] = {}
    for name, engine in self._engine_map.items():
      if isinstance(engine, LazyNativeEngine) and engine.enabled:
        try:
          await engine.ensure_loaded()
          status = engine.health_status()
          results[name] = {'loaded': True, 'device': status['device']}
        except Exception as exc:
          results[name] = {'loaded': False, 'error': str(exc)}
    return results

  @property
  def concurrency(self) -> int:
    return self._concurrency

  def close(self) -> None:
    self._store.close()


_STALE_CHECK_INTERVAL = 60  # seconds between reset_stale() calls
_STALE_THRESHOLD = 300  # jobs running longer than this are considered stale


async def run_worker_loop(
  runtime: JobRuntime,
  poll_seconds: float = 1.0,
  *,
  once: bool = False,
) -> None:
  """Main worker loop. Polls for queued jobs and processes them."""
  logger.info('Worker started, polling every %.1fs', poll_seconds)
  last_stale_check = 0.0
  while True:
    # Periodically recover jobs stuck in 'running' from crashed workers
    now = asyncio.get_event_loop().time()
    if now - last_stale_check >= _STALE_CHECK_INTERVAL:
      count = runtime._store.reset_stale(older_than_seconds=_STALE_THRESHOLD)
      if count:
        logger.info('Reset %d stale running job(s)', count)
      last_stale_check = now

    processed = await runtime.run_next()
    if once:
      return
    if not processed:
      await asyncio.sleep(poll_seconds)
