"""Integration tests for the TTS API routes: streaming, jobs, and backward compat."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.conftest import DUMMY_CHUNK, _make_config
from tts_gateway.main import create_app

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint(tmp_path) -> None:
  config = _make_config(data_dir=str(tmp_path / 'data'))
  app = create_app(config)
  with TestClient(app, raise_server_exceptions=False) as client:
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json()['ok'] is True


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
