"""Tests for JobRuntime: the single application service."""

from __future__ import annotations

import pytest

from tests.conftest import FailingEngine, MockEngine, _make_config
from tts_gateway.jobs.store import JobStore
from tts_gateway.runtime import JobRuntime, NoEnginesError, run_worker_loop


def _runtime(tmp_path, **config_overrides) -> JobRuntime:
  config_overrides.setdefault('data_dir', str(tmp_path / 'data'))
  return JobRuntime(_make_config(**config_overrides))


def _inject_engines(runtime: JobRuntime, engines: list) -> None:
  """Replace the engine map with test doubles."""
  runtime._engine_map = {e.name: e for e in engines}
  runtime._engine_chain = [e.name for e in engines]


# ---------------------------------------------------------------------------
# make_spec
# ---------------------------------------------------------------------------


def test_make_spec_defaults(tmp_path) -> None:
  rt = _runtime(tmp_path)
  spec = rt.make_spec('hello world')
  assert spec.text == 'hello world'
  assert spec.voice == ''
  assert spec.output_format == 'wav'
  assert spec.chunk_max_chars == 3000  # from _BASE_CONFIG
  assert spec.cache_namespace == 'engines=kokoro'
  rt.close()


def test_make_spec_with_voice(tmp_path) -> None:
  rt = _runtime(tmp_path, default_voice='af_heart')
  spec = rt.make_spec('hello', voice='bf_emma')
  assert spec.voice == 'bf_emma'
  rt.close()


def test_make_spec_uses_config_voice(tmp_path) -> None:
  rt = _runtime(tmp_path, default_voice='af_heart')
  spec = rt.make_spec('hello')
  assert spec.voice == 'af_heart'
  rt.close()


# ---------------------------------------------------------------------------
# submit + get
# ---------------------------------------------------------------------------


def test_submit_creates_job(tmp_path) -> None:
  rt = _runtime(tmp_path)
  spec = rt.make_spec('hello world')
  view = rt.submit(spec)
  assert view.key == spec.content_hash
  assert view.status == 'queued'
  rt.close()


def test_submit_idempotent(tmp_path) -> None:
  rt = _runtime(tmp_path)
  spec = rt.make_spec('hello world')
  v1 = rt.submit(spec)
  v2 = rt.submit(spec)
  assert v1.key == v2.key
  rt.close()


def test_get_unknown_returns_none(tmp_path) -> None:
  rt = _runtime(tmp_path)
  assert rt.get('nonexistent') is None
  rt.close()


def test_get_returns_submitted_job(tmp_path) -> None:
  rt = _runtime(tmp_path)
  spec = rt.make_spec('hello world')
  view = rt.submit(spec)
  fetched = rt.get(view.key)
  assert fetched is not None
  assert fetched.status == 'queued'
  rt.close()


# ---------------------------------------------------------------------------
# run_until_complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_until_complete_success(tmp_path) -> None:
  rt = _runtime(tmp_path)
  engine = MockEngine('mock')
  _inject_engines(rt, [engine])
  spec = rt.make_spec('hello')

  artifact = await rt.run_until_complete(spec)
  assert artifact.output_path.exists()
  assert artifact.content_type in ('audio/wav', 'audio/mpeg')

  # Job should be marked ready in store
  view = rt.get(spec.content_hash)
  assert view is not None
  assert view.status == 'ready'
  rt.close()


@pytest.mark.asyncio
async def test_run_until_complete_cached(tmp_path) -> None:
  rt = _runtime(tmp_path)
  engine = MockEngine('mock')
  _inject_engines(rt, [engine])
  spec = rt.make_spec('hello')

  a1 = await rt.run_until_complete(spec)
  calls_after_first = len(engine.calls)
  a2 = await rt.run_until_complete(spec)

  # Second call should not re-synthesize
  assert len(engine.calls) == calls_after_first
  assert a1.output_path == a2.output_path
  rt.close()


@pytest.mark.asyncio
async def test_run_until_complete_cache_partitions_engine_chain(tmp_path) -> None:
  data_dir = str(tmp_path / 'data')
  kokoro = _runtime(tmp_path, data_dir=data_dir)
  cosyvoice = _runtime(
    tmp_path,
    data_dir=data_dir,
    primary_engine='cosyvoice',
    kokoro_enabled=False,
    cosyvoice_enabled=True,
    cosyvoice_base_url='http://127.0.0.1:50000',
  )
  kokoro_engine = MockEngine('kokoro')
  cosyvoice_engine = MockEngine('cosyvoice')
  _inject_engines(kokoro, [kokoro_engine])
  _inject_engines(cosyvoice, [cosyvoice_engine])

  kokoro_artifact = await kokoro.run_until_complete(kokoro.make_spec('hello'))
  cosyvoice_artifact = await cosyvoice.run_until_complete(cosyvoice.make_spec('hello'))

  assert kokoro_artifact.request_hash != cosyvoice_artifact.request_hash
  assert kokoro_artifact.output_path != cosyvoice_artifact.output_path
  assert len(kokoro_engine.calls) == 1
  assert len(cosyvoice_engine.calls) == 1
  kokoro.close()
  cosyvoice.close()


@pytest.mark.asyncio
async def test_run_until_complete_no_engines(tmp_path) -> None:
  rt = _runtime(tmp_path, kokoro_enabled=False, pocket_enabled=False)
  spec = rt.make_spec('hello')

  with pytest.raises(NoEnginesError):
    await rt.run_until_complete(spec)
  rt.close()


@pytest.mark.asyncio
async def test_run_until_complete_engine_failure(tmp_path) -> None:
  rt = _runtime(tmp_path)
  _inject_engines(rt, [FailingEngine('bad', RuntimeError('boom'))])
  spec = rt.make_spec('hello')

  with pytest.raises(RuntimeError, match='boom'):
    await rt.run_until_complete(spec)

  # Job should be marked failed
  view = rt.get(spec.content_hash)
  assert view is not None
  assert view.status == 'failed'
  rt.close()


# ---------------------------------------------------------------------------
# run_next (worker path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_next_empty_queue(tmp_path) -> None:
  rt = _runtime(tmp_path)
  _inject_engines(rt, [MockEngine('mock')])
  assert await rt.run_next() is False
  rt.close()


@pytest.mark.asyncio
async def test_run_next_processes_job(tmp_path) -> None:
  rt = _runtime(tmp_path)
  engine = MockEngine('mock')
  _inject_engines(rt, [engine])
  spec = rt.make_spec('hello from worker')

  # Submit a job
  rt.submit(spec)

  # Worker picks it up
  assert await rt.run_next() is True

  # Job should be ready
  view = rt.get(spec.content_hash)
  assert view is not None
  assert view.status == 'ready'
  rt.close()


# ---------------------------------------------------------------------------
# Claim contention (sync vs worker)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_contention_sync_wins(tmp_path) -> None:
  """When sync claims before worker, worker gets nothing."""
  rt = _runtime(tmp_path)
  engine = MockEngine('mock')
  _inject_engines(rt, [engine])
  spec = rt.make_spec('contention test')

  # Submit
  rt.submit(spec)

  # Sync claims and executes
  artifact = await rt.run_until_complete(spec)
  assert artifact.output_path.exists()

  # Worker finds nothing to do
  assert await rt.run_next() is False
  rt.close()


@pytest.mark.asyncio
async def test_claim_contention_worker_wins(tmp_path) -> None:
  """When worker claims first, sync polls until ready."""
  rt = _runtime(tmp_path)
  engine = MockEngine('mock')
  _inject_engines(rt, [engine])
  spec = rt.make_spec('worker wins test')

  # Submit and have the worker claim it
  rt.submit(spec)
  assert await rt.run_next() is True

  # Sync request for same spec should get cached result
  artifact = await rt.run_until_complete(spec)
  assert artifact.output_path.exists()
  rt.close()


# ---------------------------------------------------------------------------
# run_worker_loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_loop_once(tmp_path) -> None:
  rt = _runtime(tmp_path)
  engine = MockEngine('mock')
  _inject_engines(rt, [engine])
  spec = rt.make_spec('worker loop test')
  rt.submit(spec)

  await run_worker_loop(rt, poll_seconds=0.1, once=True)

  view = rt.get(spec.content_hash)
  assert view is not None
  assert view.status == 'ready'
  rt.close()


@pytest.mark.asyncio
async def test_worker_loop_empty_once(tmp_path) -> None:
  rt = _runtime(tmp_path)
  _inject_engines(rt, [MockEngine('mock')])
  await run_worker_loop(rt, poll_seconds=0.1, once=True)
  rt.close()


# ---------------------------------------------------------------------------
# Store: claim(key) and reset_stale
# ---------------------------------------------------------------------------


def test_store_claim_key(tmp_path) -> None:
  store = JobStore(tmp_path / 'test.db')
  store.create_or_get('k1', '{}')

  # First claim succeeds
  assert store.claim('k1') is True
  job = store.get('k1')
  assert job is not None
  assert job.status == 'running'

  # Second claim fails (already running)
  assert store.claim('k1') is False
  store.close()


def test_store_claim_unknown_key(tmp_path) -> None:
  store = JobStore(tmp_path / 'test.db')
  assert store.claim('nonexistent') is False
  store.close()


def test_store_reset_stale(tmp_path) -> None:
  store = JobStore(tmp_path / 'test.db')
  store.create_or_get('k1', '{}')
  store.claim('k1')
  job = store.get('k1')
  assert job is not None
  assert job.status == 'running'

  # Backdate started_at (which claim() set via datetime('now')) so it's definitely stale
  store._conn.execute(
    "UPDATE jobs SET started_at = datetime(started_at, '-600 seconds') WHERE key = 'k1'"
  )
  store._conn.commit()

  count = store.reset_stale(older_than_seconds=300)
  assert count == 1
  job = store.get('k1')
  assert job is not None
  assert job.status == 'queued'
  store.close()


def test_store_create_or_get_does_not_requeue_running(tmp_path) -> None:
  """Running jobs should NOT be requeued by create_or_get."""
  store = JobStore(tmp_path / 'test.db')
  store.create_or_get('k1', '{}')
  store.claim('k1')
  job = store.get('k1')
  assert job is not None
  assert job.status == 'running'

  # Resubmit should NOT reset the running job
  job = store.create_or_get('k1', '{}')
  assert job.status == 'running'
  store.close()
