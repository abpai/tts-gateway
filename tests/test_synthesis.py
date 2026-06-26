"""Tests for the framework-free synthesis core."""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import (
  DUMMY_CHUNK,
  FailingEngine,
  FailingStreamEngine,
  MockEngine,
  MockStreamingEngine,
  SlowAfterFirstStreamEngine,
  SlowStreamEngine,
  StaggeredEngine,
)
from tts_gateway.engines.base import EngineError
from tts_gateway.render import (
  plan_chunks,
  plan_stream_chunks,
  stream_audio,
  stream_pcm,
  synthesize_chunks,
  synthesize_to_disk,
)
from tts_gateway.types import RenderPlan, SynthesisSpec

# ---------------------------------------------------------------------------
# SynthesisSpec
# ---------------------------------------------------------------------------


def test_content_hash_deterministic() -> None:
  r1 = SynthesisSpec(text='hello', voice='v', output_format='wav')
  r2 = SynthesisSpec(text='hello', voice='v', output_format='wav')
  assert r1.content_hash == r2.content_hash


def test_content_hash_varies_with_voice() -> None:
  r1 = SynthesisSpec(text='hello', voice='v1', output_format='wav')
  r2 = SynthesisSpec(text='hello', voice='v2', output_format='wav')
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_format() -> None:
  r1 = SynthesisSpec(text='hello', voice='v', output_format='wav')
  r2 = SynthesisSpec(text='hello', voice='v', output_format='mp3')
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_text() -> None:
  r1 = SynthesisSpec(text='hello', voice='v', output_format='wav')
  r2 = SynthesisSpec(text='world', voice='v', output_format='wav')
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_chunk_size() -> None:
  r1 = SynthesisSpec(text='hello', voice='v', output_format='wav', chunk_max_chars=500)
  r2 = SynthesisSpec(text='hello', voice='v', output_format='wav', chunk_max_chars=1000)
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_pipeline_version() -> None:
  r1 = SynthesisSpec(text='hello', voice='v', output_format='wav', pipeline_version='1')
  r2 = SynthesisSpec(text='hello', voice='v', output_format='wav', pipeline_version='2')
  assert r1.content_hash != r2.content_hash


def test_content_hash_varies_with_cache_namespace() -> None:
  r1 = SynthesisSpec(
    text='hello',
    voice='v',
    output_format='wav',
    cache_namespace='engines=kokoro',
  )
  r2 = SynthesisSpec(
    text='hello',
    voice='v',
    output_format='wav',
    cache_namespace='engines=cosyvoice',
  )
  assert r1.content_hash != r2.content_hash


def test_to_json_from_json_roundtrip() -> None:
  r1 = SynthesisSpec(
    text='hello',
    voice='v',
    output_format='mp3',
    chunk_max_chars=700,
    cache_namespace='engines=kokoro',
  )
  r2 = SynthesisSpec.from_json(r1.to_json())
  assert r1 == r2
  assert r1.content_hash == r2.content_hash


# ---------------------------------------------------------------------------
# plan_chunks
# ---------------------------------------------------------------------------


def test_plan_chunks_single() -> None:
  request = SynthesisSpec(text='Hello world', voice='v', output_format='wav')
  plan = plan_chunks(request)
  assert plan.chunks == ('Hello world',)
  assert plan.request_hash == request.content_hash
  assert plan.voice == 'v'


def test_plan_chunks_splits_long_text() -> None:
  request = SynthesisSpec(
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
  request = SynthesisSpec(text='   ', voice='v', output_format='wav')
  with pytest.raises(ValueError, match='empty'):
    plan_chunks(request)


# ---------------------------------------------------------------------------
# plan_stream_chunks
# ---------------------------------------------------------------------------


def test_plan_stream_chunks_smaller_first_chunk() -> None:
  text = (
    'First sentence here. Second sentence here. Third sentence here. '
    'Fourth sentence here.'
  )
  request = SynthesisSpec(
    text=text,
    voice='v',
    output_format='wav',
    chunk_max_chars=80,
  )
  disk_plan = plan_chunks(request)
  stream_plan = plan_stream_chunks(
    request,
    first_chunk_max_chars=25,
    stream_chunk_max_chars=80,
  )

  assert len(stream_plan.chunks) >= len(disk_plan.chunks)
  assert len(stream_plan.chunks[0]) <= 25
  assert len(stream_plan.chunks[0]) < len(disk_plan.chunks[0])
  assert stream_plan.request_hash == request.content_hash


def test_plan_stream_chunks_matches_disk_for_short_text() -> None:
  request = SynthesisSpec(text='Hello world', voice='v', output_format='wav')
  disk_plan = plan_chunks(request)
  stream_plan = plan_stream_chunks(
    request,
    first_chunk_max_chars=180,
    stream_chunk_max_chars=request.chunk_max_chars,
  )
  assert stream_plan.chunks == disk_plan.chunks


# ---------------------------------------------------------------------------
# synthesize_chunks — ordered parallel execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_chunks_yields_in_order() -> None:
  engine = StaggeredEngine(
    'mock',
    delays={'First.': 0.05, 'Second.': 0.0, 'Third.': 0.02},
  )
  plan = RenderPlan(
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
  plan = RenderPlan(
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
  plan = RenderPlan(
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
  plan = RenderPlan(
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
  plan = RenderPlan(
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
  plan = RenderPlan(
    request_hash='test',
    chunks=('hello',),
    voice='v',
    output_format='wav',
  )

  with pytest.raises(RuntimeError, match='no engines'):
    async for _ in synthesize_chunks(plan, [], concurrency=4, engine_timeout=10):
      pass


# ---------------------------------------------------------------------------
# stream_audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_audio_uses_stream_chunk_plan() -> None:
  text = (
    'Alpha sentence here. Beta sentence here. Gamma sentence here. Delta sentence here.'
  )
  engine = MockEngine('mock')
  request = SynthesisSpec(text=text, voice='v', output_format='wav', chunk_max_chars=80)

  async for _ in stream_audio(
    request,
    [engine],
    concurrency=4,
    engine_timeout=10,
    stream_first_chunk_max_chars=20,
    stream_chunk_max_chars=80,
  ):
    pass

  disk_first = plan_chunks(request).chunks[0]
  assert engine.calls[0][0] != disk_first
  assert len(engine.calls[0][0]) <= 20


@pytest.mark.asyncio
async def test_stream_audio_wav_yields_pcm() -> None:
  engine = MockEngine('mock')
  request = SynthesisSpec(text='hello', voice='v', output_format='wav')

  parts = []
  async for data in stream_audio(request, [engine], concurrency=4, engine_timeout=10):
    parts.append(data)

  assert len(parts) == 1
  assert parts[0] == DUMMY_CHUNK.pcm_bytes


@pytest.mark.asyncio
async def test_stream_audio_mp3_yields_encoded(monkeypatch) -> None:
  engine = MockEngine('mock')
  request = SynthesisSpec(text='hello', voice='v', output_format='mp3')

  monkeypatch.setattr(
    'tts_gateway.render.encode_output',
    lambda chunk, fmt, ffmpeg: (b'fake-mp3', 'audio/mpeg'),
  )

  parts = []
  async for data in stream_audio(request, [engine], concurrency=4, engine_timeout=10):
    parts.append(data)

  assert parts == [b'fake-mp3']


# ---------------------------------------------------------------------------
# native streaming engines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_prefers_streaming_engine_over_later_synth_only() -> None:
  streaming = MockStreamingEngine('stream')
  synth_only = MockEngine('mock')
  text = (
    'Alpha sentence here. Beta sentence here. Gamma sentence here. Delta sentence here.'
  )
  request = SynthesisSpec(text=text, voice='my-voice', output_format='wav')

  async for _ in stream_audio(request, [streaming, synth_only], engine_timeout=10):
    pass

  assert streaming.stream_calls == [(text, 'my-voice')]
  assert synth_only.calls == []


@pytest.mark.asyncio
async def test_stream_keeps_synth_primary_before_streaming_fallback() -> None:
  synth_primary = MockEngine('primary')
  streaming_fallback = MockStreamingEngine('stream')
  request = SynthesisSpec(text='hello', voice='v', output_format='wav')

  async for _ in stream_audio(
    request, [synth_primary, streaming_fallback], engine_timeout=10
  ):
    pass

  assert synth_primary.calls == [('hello', 'v')]
  assert streaming_fallback.stream_calls == []


@pytest.mark.asyncio
async def test_stream_synth_only_engines_use_chunk_planner() -> None:
  text = (
    'Alpha sentence here. Beta sentence here. Gamma sentence here. Delta sentence here.'
  )
  engine = MockEngine('mock')
  request = SynthesisSpec(text=text, voice='v', output_format='wav', chunk_max_chars=80)

  async for _ in stream_audio(
    request,
    [engine],
    engine_timeout=10,
    stream_first_chunk_max_chars=20,
    stream_chunk_max_chars=80,
  ):
    pass

  disk_first = plan_chunks(request).chunks[0]
  assert engine.calls[0][0] != disk_first
  assert len(engine.calls[0][0]) <= 20


@pytest.mark.asyncio
async def test_stream_native_receives_full_text_and_voice() -> None:
  text = 'Full request text stays intact.'
  streaming = MockStreamingEngine('stream')
  request = SynthesisSpec(text=text, voice='requested-voice', output_format='wav')

  first, rest = await stream_pcm(request, [streaming], engine_timeout=10)
  chunks = [first]
  async for chunk in rest:
    chunks.append(chunk)

  assert streaming.stream_calls == [(text, 'requested-voice')]
  assert len(chunks) == 1


@pytest.mark.asyncio
async def test_stream_fails_before_first_chunk_falls_back() -> None:
  failing = FailingStreamEngine('bad', EngineError('broken'))
  good = MockStreamingEngine('good')
  request = SynthesisSpec(text='hello', voice='v', output_format='wav')

  parts = []
  async for data in stream_audio(request, [failing, good], engine_timeout=10):
    parts.append(data)

  assert parts == [DUMMY_CHUNK.pcm_bytes]
  assert good.stream_calls == [('hello', 'v')]


@pytest.mark.asyncio
async def test_stream_fails_after_first_chunk_propagates() -> None:
  failing = FailingStreamEngine('bad', EngineError('mid-stream'), chunks_before_error=1)
  request = SynthesisSpec(text='hello', voice='v', output_format='wav')

  stream = stream_audio(request, [failing], engine_timeout=10)
  first = await stream.__anext__()
  assert first == DUMMY_CHUNK.pcm_bytes

  with pytest.raises(EngineError, match='mid-stream'):
    await stream.__anext__()


@pytest.mark.asyncio
async def test_stream_timeout_raises_instead_of_hanging() -> None:
  slow = SlowStreamEngine('slow', delay=0.2)
  request = SynthesisSpec(text='hello', voice='v', output_format='wav')

  with pytest.raises(asyncio.TimeoutError):
    async for _ in stream_audio(request, [slow], engine_timeout=0.05):
      pass


@pytest.mark.asyncio
async def test_stream_timeout_after_first_chunk_raises() -> None:
  slow = SlowAfterFirstStreamEngine('slow', delay=0.2)
  request = SynthesisSpec(text='hello', voice='v', output_format='wav')
  stream = stream_audio(request, [slow], engine_timeout=0.05)

  first = await stream.__anext__()
  assert first == DUMMY_CHUNK.pcm_bytes

  with pytest.raises(asyncio.TimeoutError):
    await stream.__anext__()


@pytest.mark.asyncio
async def test_synthesize_to_disk_ignores_streaming_protocol(tmp_path) -> None:
  streaming = MockStreamingEngine('stream')
  request = SynthesisSpec(text='Hello world', voice='v', output_format='wav')

  artifact = await synthesize_to_disk(
    request, [streaming], tmp_path, concurrency=4, engine_timeout=10
  )

  assert artifact.output_path.exists()
  assert streaming.stream_calls == []
  assert streaming.synth_calls == [('Hello world', 'v')]


# ---------------------------------------------------------------------------
# synthesize_to_disk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_to_disk_creates_files(tmp_path) -> None:
  engine = MockEngine('mock')
  request = SynthesisSpec(text='Hello world', voice='v', output_format='wav')

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
  request = SynthesisSpec(text='Hello world', voice='v', output_format='wav')

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
  r1 = SynthesisSpec(text='Hello', voice='v', output_format='wav')
  r2 = SynthesisSpec(text='World', voice='v', output_format='wav')

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
  request = SynthesisSpec(
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
