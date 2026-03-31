"""Tests for the async job worker."""

from __future__ import annotations

import pytest

from tests.conftest import FailingEngine, MockEngine
from tts_gateway.jobs.store import JobStore
from tts_gateway.jobs.worker import process_job, run_worker
from tts_gateway.synthesis import SynthesisRequest


@pytest.fixture()
def store(tmp_path) -> JobStore:
  s = JobStore(tmp_path / 'test.db')
  yield s
  s.close()


@pytest.fixture()
def artifacts_dir(tmp_path):
  d = tmp_path / 'artifacts'
  d.mkdir()
  return d


def _make_request_json(**overrides) -> str:
  request = SynthesisRequest(
    text=overrides.get('text', 'Hello world'),
    voice=overrides.get('voice', 'v'),
    output_format=overrides.get('output_format', 'wav'),
  )
  return request.to_json()


@pytest.mark.asyncio
async def test_process_job_success(store: JobStore, artifacts_dir) -> None:
  request_json = _make_request_json()
  store.create_or_get('key1', request_json)
  store.claim_next()

  await process_job(
    store,
    'key1',
    request_json,
    [MockEngine()],
    artifacts_dir,
    concurrency=4,
    engine_timeout=10,
  )

  job = store.get('key1')
  assert job is not None
  assert job.status == 'ready'
  assert job.artifact_path is not None
  assert job.chunks_total == 1


@pytest.mark.asyncio
async def test_process_job_failure(store: JobStore, artifacts_dir) -> None:
  request_json = _make_request_json()
  store.create_or_get('key1', request_json)
  store.claim_next()

  await process_job(
    store,
    'key1',
    request_json,
    [FailingEngine('fail', RuntimeError('engine exploded'))],
    artifacts_dir,
    concurrency=4,
    engine_timeout=10,
  )

  job = store.get('key1')
  assert job is not None
  assert job.status == 'failed'
  assert 'engine exploded' in (job.error or '')


@pytest.mark.asyncio
async def test_run_worker_once(store: JobStore, artifacts_dir) -> None:
  request_json = _make_request_json()
  store.create_or_get('key1', request_json)

  await run_worker(
    store,
    [MockEngine()],
    artifacts_dir,
    once=True,
    concurrency=4,
    engine_timeout=10,
  )

  job = store.get('key1')
  assert job is not None
  assert job.status == 'ready'


@pytest.mark.asyncio
async def test_run_worker_once_empty_queue(store: JobStore, artifacts_dir) -> None:
  await run_worker(
    store,
    [MockEngine()],
    artifacts_dir,
    once=True,
    concurrency=4,
    engine_timeout=10,
  )
