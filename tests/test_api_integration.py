"""Integration tests for the TTS API routes: streaming, jobs, and backward compat."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from starlette.responses import Response

from tests.conftest import DUMMY_CHUNK, _make_config
from tts_gateway.engines.base import AudioChunk
from tts_gateway.main import create_app
from tts_gateway.render import plan_chunks, plan_stream_chunks
from tts_gateway.routes import abort_on_client_disconnect
from tts_gateway.types import SynthesisSpec

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint(tmp_path) -> None:
  config = _make_config(
    data_dir=str(tmp_path / 'data'),
    stream_first_chunk_max_chars=150,
    stream_chunk_max_chars=400,
  )
  app = create_app(config)
  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.get('/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body['ok'] is True
    assert body['streamFirstChunkMaxChars'] == 150
    assert body['streamChunkMaxChars'] == 400


# ---------------------------------------------------------------------------
# Legacy buffered TTS (POST /tts with Accept: audio/*)
# ---------------------------------------------------------------------------


def test_tts_legacy_audio_accept(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch(
      'tts_gateway.render.encode_output', return_value=(b'wav-data', 'audio/wav')
    ):
      with patch('tts_gateway.render._try_engines', return_value=DUMMY_CHUNK):
        resp = client.post(
          '/tts',
          data={'text': 'Hello world'},
          headers={'Accept': 'audio/wav'},
        )

    assert resp.status_code == 200
    assert resp.content == b'wav-data'


def test_tts_default_accept_returns_audio(tmp_path) -> None:
  """POST /tts without explicit Accept header should return audio (backward compat)."""
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch(
      'tts_gateway.render.encode_output', return_value=(b'wav-data', 'audio/wav')
    ):
      with patch('tts_gateway.render._try_engines', return_value=DUMMY_CHUNK):
        resp = client.post('/tts', data={'text': 'Hello world'})

    assert resp.status_code == 200
    assert resp.content == b'wav-data'


# ---------------------------------------------------------------------------
# Legacy buffered TTS at /tts/sync
# ---------------------------------------------------------------------------


def test_tts_sync_endpoint(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch(
      'tts_gateway.render.encode_output', return_value=(b'wav-data', 'audio/wav')
    ):
      with patch('tts_gateway.render._try_engines', return_value=DUMMY_CHUNK):
        resp = client.post('/tts/sync', data={'text': 'Hello world'})

    assert resp.status_code == 200
    assert resp.content == b'wav-data'


# ---------------------------------------------------------------------------
# Job-based TTS (POST /tts with Accept: application/json)
# ---------------------------------------------------------------------------


def test_tts_job_submit(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.post(
      '/tts',
      data={'text': 'Hello world'},
      headers={'Accept': 'application/json'},
    )

    assert resp.status_code == 202
    data = resp.json()
    assert data['status'] == 'queued'
    assert 'key' in data
    assert data['key']


def test_tts_job_submit_idempotent(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    r1 = client.post(
      '/tts',
      data={'text': 'Same text'},
      headers={'Accept': 'application/json'},
    )
    r2 = client.post(
      '/tts',
      data={'text': 'Same text'},
      headers={'Accept': 'application/json'},
    )

    assert r1.json()['key'] == r2.json()['key']


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


def test_tts_job_status_not_found(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.get('/tts/nonexistent')
    assert resp.status_code == 404


def test_tts_job_status_queued(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    submit = client.post(
      '/tts',
      data={'text': 'Hello'},
      headers={'Accept': 'application/json'},
    )
    key = submit.json()['key']

    resp = client.get(f'/tts/{key}')
    assert resp.status_code in (200, 202)
    assert resp.json()['status'] in ('queued', 'running', 'ready')


# ---------------------------------------------------------------------------
# Job audio download
# ---------------------------------------------------------------------------


def test_tts_job_audio_not_ready(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    submit = client.post(
      '/tts',
      data={'text': 'Hello'},
      headers={'Accept': 'application/json'},
    )
    key = submit.json()['key']

    resp = client.get(f'/tts/{key}/audio')
    assert resp.status_code in (200, 409)


def test_tts_job_audio_not_found(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.get('/tts/nonexistent/audio')
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Streaming TTS
# ---------------------------------------------------------------------------


def test_tts_stream_empty_text(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.post('/tts/stream', json={'text': '  '})
    assert resp.status_code == 422


def test_tts_stream_returns_mpeg(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch('tts_gateway.render.encode_output', return_value=(b'mp3', 'audio/mpeg')):
      with patch('tts_gateway.render._try_engines', return_value=DUMMY_CHUNK):
        resp = client.post('/tts/stream', json={'text': 'Hello stream'})

    assert resp.status_code == 200
    assert resp.headers['content-type'].startswith('audio/mpeg')
    assert resp.headers['x-tts-mode'] == 'stream'


def test_tts_stream_no_engines(tmp_path) -> None:
  config = _make_config(
    data_dir=str(tmp_path / 'data'),
    kokoro_enabled=False,
    pocket_enabled=False,
  )
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.post('/tts/stream', json={'text': 'Hello stream'})
    assert resp.status_code == 503


def test_tts_stream_first_audio_timeout_returns_504(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  async def timeout_stream() -> AsyncGenerator[bytes, None]:
    if False:
      yield b''
    raise TimeoutError('slow stream')

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch('tts_gateway.routes.stream_audio', return_value=timeout_stream()):
      resp = client.post('/tts/stream', json={'text': 'Hello stream'})

  assert resp.status_code == 504
  assert resp.json()['error'] == 'stream first audio timed out'


def test_tts_stream_pcm_empty_text(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.post('/tts/stream/pcm', json={'text': '  '})
    assert resp.status_code == 422


def test_tts_stream_pcm_no_engines(tmp_path) -> None:
  config = _make_config(
    data_dir=str(tmp_path / 'data'),
    kokoro_enabled=False,
    pocket_enabled=False,
  )
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.post('/tts/stream/pcm', json={'text': 'Hello pcm'})
    assert resp.status_code == 503


def test_tts_stream_pcm_first_audio_timeout_returns_504(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch(
      'tts_gateway.routes.stream_pcm',
      AsyncMock(side_effect=TimeoutError('slow stream')),
    ):
      resp = client.post('/tts/stream/pcm', json={'text': 'Hello pcm'})

  assert resp.status_code == 504
  assert resp.json()['error'] == 'stream first audio timed out'


def test_tts_stream_pcm_returns_raw_pcm_headers(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch('tts_gateway.render._try_engines', return_value=DUMMY_CHUNK):
      resp = client.post('/tts/stream/pcm', json={'text': 'Hello pcm'})

    assert resp.status_code == 200
    assert resp.headers['content-type'].startswith('audio/raw')
    assert resp.headers['x-tts-mode'] == 'stream-pcm'
    assert resp.headers['x-tts-primary-engine'] == config.primary_engine
    assert resp.headers['x-tts-sample-rate'] == str(DUMMY_CHUNK.sample_rate)
    assert resp.headers['x-tts-channels'] == str(DUMMY_CHUNK.channels)
    assert resp.headers['x-tts-sample-width'] == str(DUMMY_CHUNK.sample_width)
    assert resp.headers['x-tts-pcm-format'] == 's16le'
    assert resp.content == DUMMY_CHUNK.pcm_bytes


def test_tts_stream_pcm_unsupported_sample_width_returns_502(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)
  chunk = AudioChunk(
    pcm_bytes=b'\x00\x00\x00',
    sample_rate=24_000,
    channels=1,
    sample_width=3,
  )

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch('tts_gateway.render._try_engines', return_value=chunk):
      resp = client.post('/tts/stream/pcm', json={'text': 'Hello pcm'})

    assert resp.status_code == 502
    assert 'unsupported sample width' in resp.json()['error']


def test_tts_stream_pcm_uses_stream_chunk_plan(tmp_path) -> None:
  config = _make_config(
    data_dir=str(tmp_path / 'data'),
    chunk_max_chars=80,
    stream_first_chunk_max_chars=20,
    stream_chunk_max_chars=80,
  )
  app = create_app(config)
  text = (
    'Alpha sentence here. Beta sentence here. Gamma sentence here. Delta sentence here.'
  )
  spec = SynthesisSpec(text=text, voice='', output_format='wav', chunk_max_chars=80)
  stream_first = plan_stream_chunks(
    spec,
    first_chunk_max_chars=20,
    stream_chunk_max_chars=80,
  ).chunks[0]
  recorded: list[str] = []

  async def capture_try_engines(
    synth_text: str,
    voice: str,
    engines: list,
    *,
    timeout: float,
  ):
    recorded.append(synth_text)
    return DUMMY_CHUNK

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch('tts_gateway.render._try_engines', side_effect=capture_try_engines):
      resp = client.post('/tts/stream/pcm', json={'text': text})

    assert resp.status_code == 200
    assert stream_first in recorded
    assert len(stream_first) <= 20
    disk_first = plan_chunks(spec).chunks[0]
    assert stream_first != disk_first


def test_tts_stream_uses_stream_chunk_plan(tmp_path) -> None:
  config = _make_config(
    data_dir=str(tmp_path / 'data'),
    chunk_max_chars=80,
    stream_first_chunk_max_chars=20,
    stream_chunk_max_chars=80,
  )
  app = create_app(config)
  text = (
    'Alpha sentence here. Beta sentence here. Gamma sentence here. Delta sentence here.'
  )
  spec = SynthesisSpec(text=text, voice='', output_format='wav', chunk_max_chars=80)
  stream_first = plan_stream_chunks(
    spec,
    first_chunk_max_chars=20,
    stream_chunk_max_chars=80,
  ).chunks[0]
  recorded: list[str] = []

  async def capture_try_engines(
    synth_text: str,
    voice: str,
    engines: list,
    *,
    timeout: float,
  ):
    recorded.append(synth_text)
    return DUMMY_CHUNK

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch('tts_gateway.render.encode_output', return_value=(b'mp3', 'audio/mpeg')):
      with patch('tts_gateway.render._try_engines', side_effect=capture_try_engines):
        resp = client.post('/tts/stream', json={'text': text})

    assert resp.status_code == 200
    assert stream_first in recorded
    assert len(stream_first) <= 20
    disk_first = plan_chunks(spec).chunks[0]
    assert stream_first != disk_first


async def test_abort_on_client_disconnect_returns_none_when_connected() -> None:
  request = MagicMock()
  request.is_disconnected = AsyncMock(return_value=False)
  assert await abort_on_client_disconnect(request) is None


async def test_abort_on_client_disconnect_closes_stream() -> None:
  request = MagicMock()
  request.is_disconnected = AsyncMock(return_value=True)
  stream = MagicMock()
  stream.aclose = AsyncMock()
  response = await abort_on_client_disconnect(request, stream=stream)
  assert response is not None
  assert response.status_code == 499
  stream.aclose.assert_awaited_once()


def test_tts_stream_returns_499_when_disconnected_before_prefetch(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with patch(
    'tts_gateway.routes.abort_on_client_disconnect',
    new=AsyncMock(return_value=Response(status_code=499)),
  ):
    with patch('tts_gateway.routes.stream_audio') as stream_audio:
      with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post('/tts/stream', json={'text': 'Hello stream'})

  assert resp.status_code == 499
  stream_audio.assert_not_called()


def test_tts_stream_pcm_returns_499_when_disconnected_after_prefetch(
  tmp_path,
) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)
  pcm_stream = MagicMock()
  pcm_stream.aclose = AsyncMock()

  async def fake_stream_pcm(*args, **kwargs):
    return DUMMY_CHUNK, pcm_stream

  disconnect = AsyncMock(side_effect=[None, Response(status_code=499)])

  with patch('tts_gateway.routes.abort_on_client_disconnect', new=disconnect):
    with patch('tts_gateway.routes.stream_pcm', side_effect=fake_stream_pcm):
      with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post('/tts/stream/pcm', json={'text': 'Hello pcm'})

  assert resp.status_code == 499
  assert disconnect.await_count >= 2
  assert disconnect.await_args_list[1].kwargs['stream'] is pcm_stream


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_tts_empty_text(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.post('/tts', data={'text': ''})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/ canonical routes
# ---------------------------------------------------------------------------


def test_v1_speech(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    with patch(
      'tts_gateway.render.encode_output', return_value=(b'wav-data', 'audio/wav')
    ):
      with patch('tts_gateway.render._try_engines', return_value=DUMMY_CHUNK):
        resp = client.post('/v1/speech', data={'text': 'Hello v1'})

    assert resp.status_code == 200
    assert resp.content == b'wav-data'


def test_v1_jobs_submit(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.post('/v1/jobs', data={'text': 'Hello v1 jobs'})
    assert resp.status_code == 202
    data = resp.json()
    assert data['status'] == 'queued'
    assert 'key' in data


def test_v1_jobs_status(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    submit = client.post('/v1/jobs', data={'text': 'Hello v1 status'})
    key = submit.json()['key']

    resp = client.get(f'/v1/jobs/{key}')
    assert resp.status_code in (200, 202)
    assert resp.json()['key'] == key


def test_v1_jobs_status_not_found(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.get('/v1/jobs/nonexistent')
    assert resp.status_code == 404


def test_v1_jobs_audio_not_found(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)

  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.get('/v1/jobs/nonexistent/audio')
    assert resp.status_code == 404
