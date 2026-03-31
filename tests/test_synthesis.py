"""Tests for the framework-free synthesis core."""

from __future__ import annotations

import pytest

from tests.conftest import DUMMY_CHUNK, FailingEngine, MockEngine, StaggeredEngine
from tts_gateway.engines.base import EngineError
from tts_gateway.synthesis import (
  ChunkPlan,
  SynthesisRequest,
  plan_chunks,
  stream_synthesis,
  synthesize_chunks,
  synthesize_to_disk,
)

# ---------------------------------------------------------------------------
# SynthesisRequest
# ---------------------------------------------------------------------------


def test_content_hash_deterministic() -> None:
  r1 = SynthesisRequest(text='hello', voice='v', output_format='wav')
  r2 = SynthesisRequest(text='hello', voice='v', output_format='wav')
  assert r1.content_hash == r2.content_hash


def test_content_hash_varies_with_voice() -> None:
  r1 = SynthesisRequest(text='hello', voice='v1', output_format='wav')
  r2 = SynthesisRequest(text='hello', voice='v2', output_format='wav')
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_format() -> None:
  r1 = SynthesisRequest(text='hello', voice='v', output_format='wav')
  r2 = SynthesisRequest(text='hello', voice='v', output_format='mp3')
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_text() -> None:
  r1 = SynthesisRequest(text='hello', voice='v', output_format='wav')
  r2 = SynthesisRequest(text='world', voice='v', output_format='wav')
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_chunk_size() -> None:
  r1 = SynthesisRequest(
    text='hello', voice='v', output_format='wav', chunk_max_chars=500
  )
  r2 = SynthesisRequest(
    text='hello', voice='v', output_format='wav', chunk_max_chars=1000
  )
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_pipeline_version() -> None:
  r1 = SynthesisRequest(
    text='hello', voice='v', output_format='wav', pipeline_version='1'
  )
  r2 = SynthesisRequest(
    text='hello', voice='v', output_format='wav', pipeline_version='2'
  )
  assert r1.content_hash != r2.content_hash


def test_to_json_from_json_roundtrip() -> None:
  r1 = SynthesisRequest(
    text='hello', voice='v', output_format='mp3', chunk_max_chars=700
  )
  r2 = SynthesisRequest.from_json(r1.to_json())
  assert r1 == r2
  assert r1.content_hash == r2.content_hash


# ---------------------------------------------------------------------------
# plan_chunks
# ---------------------------------------------------------------------------


def test_plan_chunks_single() -> None:
  request = SynthesisRequest(text='Hello world', voice='v', output_format='wav')
  plan = plan_chunks(request)
  assert plan.chunks == ('Hello world',)
  assert plan.request_hash == request.content_hash
  assert plan.voice == 'v'


def test_plan_chunks_splits_long_text() -> None:
  request = SynthesisRequest(
    text='First sentence. Second sentence. Third sentence.',
    voice='v',
    output_format='wav',
    chunk_max_chars=20,
  )
  plan = plan_chunks(request)
  assert len(plan.chunks) > 1
  for chunk in plan.chunks:
    assert len(chunk) <= 20


def test_plan_chunks_empty_raises() -> None:
  request = SynthesisRequest(text='   ', voice='v', output_format='wav')
  with pytest.raises(ValueError, match='empty'):
    plan_chunks(request)


# ---------------------------------------------------------------------------
# synthesize_chunks — ordered parallel execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_chunks_yields_in_order() -> None:
  engine = StaggeredEngine(
    'mock',
    delays={'First.': 0.05, 'Second.': 0.0, 'Third.': 0.02},
  )
  plan = ChunkPlan(
    request_hash='test',
    chunks=('First.', 'Second.', 'Third.'),
    voice='v',
    output_format='wav',
  )

  chunks = []
  async for chunk in synthesize_chunks(
    plan, [engine], concurrency=4, engine_timeout=10
  ):
    chunks.append(chunk)

  assert [c.pcm_bytes for c in chunks] == [b'First.', b'Second.', b'Third.']


@pytest.mark.asyncio
async def test_synthesize_chunks_runs_concurrently() -> None:
  engine = StaggeredEngine(
    'mock',
    delays={'a': 0.05, 'b': 0.05, 'c': 0.05},
  )
  plan = ChunkPlan(
    request_hash='test',
    chunks=('a', 'b', 'c'),
    voice='v',
    output_format='wav',
  )

  async for _ in synthesize_chunks(plan, [engine], concurrency=4, engine_timeout=10):
    pass

  assert engine.max_active_calls >= 2


@pytest.mark.asyncio
async def test_synthesize_chunks_respects_concurrency() -> None:
  engine = StaggeredEngine(
    'mock',
    delays={'a': 0.05, 'b': 0.05, 'c': 0.05, 'd': 0.05},
  )
  plan = ChunkPlan(
    request_hash='test',
    chunks=('a', 'b', 'c', 'd'),
    voice='v',
    output_format='wav',
  )

  async for _ in synthesize_chunks(plan, [engine], concurrency=2, engine_timeout=10):
    pass

  assert engine.max_active_calls <= 2


@pytest.mark.asyncio
async def test_synthesize_chunks_engine_fallback() -> None:
  failing = FailingEngine('bad', EngineError('broken'))
  good = MockEngine('good')
  plan = ChunkPlan(
    request_hash='test',
    chunks=('hello',),
    voice='v',
    output_format='wav',
  )

  chunks = []
  async for chunk in synthesize_chunks(
    plan, [failing, good], concurrency=4, engine_timeout=10
  ):
    chunks.append(chunk)

  assert len(chunks) == 1
  assert good.calls == [('hello', 'v')]


@pytest.mark.asyncio
async def test_synthesize_chunks_all_engines_fail() -> None:
  failing = FailingEngine('bad', EngineError('broken'))
  plan = ChunkPlan(
    request_hash='test',
    chunks=('hello',),
    voice='v',
    output_format='wav',
  )

  with pytest.raises(RuntimeError, match='all engines failed'):
    async for _ in synthesize_chunks(plan, [failing], concurrency=4, engine_timeout=10):
      pass


@pytest.mark.asyncio
async def test_synthesize_chunks_no_engines() -> None:
  plan = ChunkPlan(
    request_hash='test',
    chunks=('hello',),
    voice='v',
    output_format='wav',
  )

  with pytest.raises(RuntimeError, match='no engines'):
    async for _ in synthesize_chunks(plan, [], concurrency=4, engine_timeout=10):
      pass


# ---------------------------------------------------------------------------
# stream_synthesis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_synthesis_wav_yields_pcm() -> None:
  engine = MockEngine('mock')
  request = SynthesisRequest(text='hello', voice='v', output_format='wav')

  parts = []
  async for data in stream_synthesis(
    request, [engine], concurrency=4, engine_timeout=10
  ):
    parts.append(data)

  assert len(parts) == 1
  assert parts[0] == DUMMY_CHUNK.pcm_bytes


@pytest.mark.asyncio
async def test_stream_synthesis_mp3_yields_encoded(monkeypatch) -> None:
  engine = MockEngine('mock')
  request = SynthesisRequest(text='hello', voice='v', output_format='mp3')

  monkeypatch.setattr(
    'tts_gateway.synthesis.encode_output',
    lambda chunk, fmt, ffmpeg: (b'fake-mp3', 'audio/mpeg'),
  )

  parts = []
  async for data in stream_synthesis(
    request, [engine], concurrency=4, engine_timeout=10
  ):
    parts.append(data)

  assert parts == [b'fake-mp3']


# ---------------------------------------------------------------------------
# synthesize_to_disk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_to_disk_creates_files(tmp_path) -> None:
  engine = MockEngine('mock')
  request = SynthesisRequest(text='Hello world', voice='v', output_format='wav')

  artifact = await synthesize_to_disk(
    request, [engine], tmp_path, concurrency=4, engine_timeout=10
  )

  assert artifact.output_path.exists()
  assert artifact.content_type == 'audio/wav'
  assert artifact.chunks_total == 1
  assert artifact.duration_ms >= 0

  chunk_wav = tmp_path / artifact.request_hash / 'chunk_000.wav'
  assert chunk_wav.exists()


@pytest.mark.asyncio
async def test_synthesize_to_disk_cache_hit(tmp_path) -> None:
  engine = MockEngine('mock')
  request = SynthesisRequest(text='Hello world', voice='v', output_format='wav')

  a1 = await synthesize_to_disk(
    request, [engine], tmp_path, concurrency=4, engine_timeout=10
  )
  calls_after_first = len(engine.calls)

  a2 = await synthesize_to_disk(
    request, [engine], tmp_path, concurrency=4, engine_timeout=10
  )

  assert a1.output_path == a2.output_path
  assert a2.duration_ms == 0
  assert len(engine.calls) == calls_after_first


@pytest.mark.asyncio
async def test_synthesize_to_disk_content_addressed(tmp_path) -> None:
  engine = MockEngine('mock')
  r1 = SynthesisRequest(text='Hello', voice='v', output_format='wav')
  r2 = SynthesisRequest(text='World', voice='v', output_format='wav')

  a1 = await synthesize_to_disk(
    r1, [engine], tmp_path, concurrency=4, engine_timeout=10
  )
  a2 = await synthesize_to_disk(
    r2, [engine], tmp_path, concurrency=4, engine_timeout=10
  )

  assert a1.request_hash != a2.request_hash
  assert a1.output_path != a2.output_path


@pytest.mark.asyncio
async def test_synthesize_to_disk_resumes_partial(tmp_path) -> None:
  """If some chunk WAVs exist on disk, only synthesize the missing ones."""
  engine = MockEngine('mock')
  request = SynthesisRequest(
    text='First sentence. Second sentence. Third sentence.',
    voice='v',
    output_format='wav',
    chunk_max_chars=20,
  )

  a1 = await synthesize_to_disk(
    request, [engine], tmp_path, concurrency=4, engine_timeout=10
  )
  first_call_count = len(engine.calls)
  assert first_call_count > 1

  a1.output_path.unlink()

  a2 = await synthesize_to_disk(
    request, [engine], tmp_path, concurrency=4, engine_timeout=10
  )
  assert a2.output_path.exists()
  assert len(engine.calls) == first_call_count
