"""Tests for the SQLite job store."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tts_gateway.jobs.store import JobStore


@pytest.fixture()
def store(tmp_path) -> Iterator[JobStore]:
  s = JobStore(tmp_path / 'test.db')
  yield s
  s.close()


def test_create_and_get(store: JobStore) -> None:
  record = store.create_or_get('key1', '{"text":"hello"}')
  assert record.key == 'key1'
  assert record.status == 'queued'
  assert record.request_json == '{"text":"hello"}'
  assert record.chunks_done == 0
  assert record.created_at is not None


def test_create_idempotent(store: JobStore) -> None:
  r1 = store.create_or_get('key1', '{"text":"hello"}')
  r2 = store.create_or_get('key1', '{"text":"different"}')
  # Second create should return the existing record unchanged
  assert r1.key == r2.key
  assert r2.request_json == '{"text":"hello"}'


def test_get_unknown_returns_none(store: JobStore) -> None:
  assert store.get('nonexistent') is None


def test_claim_next_empty_returns_none(store: JobStore) -> None:
  assert store.claim_next() is None


def test_claim_next_returns_oldest_queued(store: JobStore) -> None:
  store.create_or_get('key1', '{}')
  store.create_or_get('key2', '{}')

  claimed = store.claim_next()
  assert claimed is not None
  assert claimed.key == 'key1'
  assert claimed.status == 'running'
  assert claimed.started_at is not None

  # key1 is now running, so claim_next should return key2
  claimed2 = store.claim_next()
  assert claimed2 is not None
  assert claimed2.key == 'key2'

  # No more queued jobs
  assert store.claim_next() is None


def test_update_progress(store: JobStore) -> None:
  store.create_or_get('key1', '{}')
  store.update_progress('key1', chunks_done=3, chunks_total=10)

  record = store.get('key1')
  assert record is not None
  assert record.chunks_done == 3
  assert record.chunks_total == 10


def test_mark_encoding(store: JobStore) -> None:
  store.create_or_get('key1', '{}')
  store.mark_encoding('key1')

  record = store.get('key1')
  assert record is not None
  assert record.status == 'encoding'


def test_mark_ready(store: JobStore) -> None:
  store.create_or_get('key1', '{}')
  store.mark_ready(
    'key1',
    artifact_path='/tmp/output.mp3',
    content_type='audio/mpeg',
    chunks_total=5,
  )

  record = store.get('key1')
  assert record is not None
  assert record.status == 'ready'
  assert record.artifact_path == '/tmp/output.mp3'
  assert record.content_type == 'audio/mpeg'
  assert record.chunks_total == 5
  assert record.chunks_done == 5
  assert record.completed_at is not None


def test_mark_failed(store: JobStore) -> None:
  store.create_or_get('key1', '{}')
  store.mark_failed('key1', 'engine crashed')

  record = store.get('key1')
  assert record is not None
  assert record.status == 'failed'
  assert record.error == 'engine crashed'
  assert record.completed_at is not None


def test_list_jobs(store: JobStore) -> None:
  store.create_or_get('key1', '{}')
  store.create_or_get('key2', '{}')
  store.mark_ready(
    'key1', artifact_path='/tmp/a.wav', content_type='audio/wav', chunks_total=1
  )

  all_jobs = store.list_jobs()
  assert len(all_jobs) == 2

  ready_jobs = store.list_jobs(status='ready')
  assert len(ready_jobs) == 1
  assert ready_jobs[0].key == 'key1'

  queued_jobs = store.list_jobs(status='queued')
  assert len(queued_jobs) == 1
  assert queued_jobs[0].key == 'key2'


def test_failed_job_requeued_on_resubmit(store: JobStore) -> None:
  """Resubmitting a failed job should reset it to queued."""
  store.create_or_get('key1', '{}')
  store.claim_next()
  store.mark_failed('key1', 'engine crashed')

  job = store.get('key1')
  assert job is not None
  assert job.status == 'failed'

  # Resubmit same key
  job2 = store.create_or_get('key1', '{}')
  assert job2.status == 'queued'
  assert job2.error is None
  assert job2.started_at is None
  assert job2.completed_at is None


def test_full_lifecycle(store: JobStore) -> None:
  """Test the complete job lifecycle: queued → running → encoding → ready."""
  store.create_or_get('key1', '{"text":"hello"}')

  # Claim
  job = store.claim_next()
  assert job is not None
  assert job.status == 'running'

  # Progress
  store.update_progress('key1', chunks_done=1, chunks_total=3)
  store.update_progress('key1', chunks_done=2, chunks_total=3)

  # Encoding
  store.mark_encoding('key1')
  job = store.get('key1')
  assert job is not None
  assert job.status == 'encoding'

  # Ready
  store.mark_ready(
    'key1',
    artifact_path='/data/output.mp3',
    content_type='audio/mpeg',
    chunks_total=3,
  )
  job = store.get('key1')
  assert job is not None
  assert job.status == 'ready'
  assert job.chunks_done == 3
