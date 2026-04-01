"""Tests for worker execution via JobRuntime.run_next() and run_worker_loop()."""

from __future__ import annotations

import pytest

from tests.conftest import FailingEngine, MockEngine, _make_config
from tts_gateway.runtime import JobRuntime, run_worker_loop


def _runtime(tmp_path, **overrides) -> JobRuntime:
  overrides.setdefault('data_dir', str(tmp_path / 'data'))
  return JobRuntime(_make_config(**overrides))


def _inject_engines(rt: JobRuntime, engines: list) -> None:
  rt._engine_map = {e.name: e for e in engines}
  rt._engine_chain = [e.name for e in engines]


@pytest.mark.asyncio
async def test_process_job_success(tmp_path) -> None:
  rt = _runtime(tmp_path)
  _inject_engines(rt, [MockEngine()])
  spec = rt.make_spec('Hello world')
  rt.submit(spec)

  assert await rt.run_next() is True

  view = rt.get(spec.content_hash)
  assert view is not None
  assert view.status == 'ready'
  assert view.chunks_total == 1
  rt.close()


@pytest.mark.asyncio
async def test_process_job_failure(tmp_path) -> None:
  rt = _runtime(tmp_path)
  _inject_engines(rt, [FailingEngine('fail', RuntimeError('engine exploded'))])
  spec = rt.make_spec('Hello world')
  rt.submit(spec)

  assert await rt.run_next() is True

  view = rt.get(spec.content_hash)
  assert view is not None
  assert view.status == 'failed'
  assert 'engine exploded' in (view.error or '')
  rt.close()


@pytest.mark.asyncio
async def test_run_worker_once(tmp_path) -> None:
  rt = _runtime(tmp_path)
  _inject_engines(rt, [MockEngine()])
  spec = rt.make_spec('Hello world')
  rt.submit(spec)

  await run_worker_loop(rt, poll_seconds=0.1, once=True)

  view = rt.get(spec.content_hash)
  assert view is not None
  assert view.status == 'ready'
  rt.close()


@pytest.mark.asyncio
async def test_run_worker_once_empty_queue(tmp_path) -> None:
  rt = _runtime(tmp_path)
  _inject_engines(rt, [MockEngine()])
  await run_worker_loop(rt, poll_seconds=0.1, once=True)
  rt.close()
