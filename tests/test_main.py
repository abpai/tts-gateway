from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import _make_config
from tts_gateway.main import create_app
from tts_gateway.speech_cli import HealthReport
from tts_gateway.version import package_version


def test_health_reports_chunk_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr('tts_gateway.runtime.os.cpu_count', lambda: 8)
  app = create_app(_make_config(kokoro_enabled=False, chunk_max_chars=3000))
  client = TestClient(app)

  response = client.get('/health')

  assert response.status_code == 200
  assert response.json()['chunkConcurrency'] == 4
  assert response.json()['packageVersion'] == package_version()
  assert response.json()['deviceMode'] == 'cpu'
  assert 'defaultSpeed' not in response.json()


def test_health_ok_false_when_no_engines_available() -> None:
  app = create_app(
    _make_config(kokoro_enabled=False, pocket_enabled=False, cosyvoice_enabled=False)
  )
  client = TestClient(app)

  response = client.get('/health')

  assert response.status_code == 200
  assert response.json()['ok'] is False


def test_health_ok_true_when_an_engine_is_available() -> None:
  app = create_app(_make_config(kokoro_enabled=True))
  client = TestClient(app)

  response = client.get('/health')

  assert response.status_code == 200
  assert response.json()['ok'] is True


def test_health_matches_cli_contract() -> None:
  """Locks the server/CLI contract: /health must validate as a HealthReport."""
  app = create_app(_make_config(kokoro_enabled=True))
  client = TestClient(app)

  response = client.get('/health')

  assert response.status_code == 200
  HealthReport.model_validate(response.json())


def test_tts_stream_accepts_cli_payload_shape() -> None:
  """POST /tts/stream must accept the JSON body shape the CLI sends."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/tts/stream',
    json={'text': 'hello world', 'voice': 'af_heart', 'speed': 1.0},
  )

  assert response.status_code != 422
  if response.status_code >= 400:
    assert 'error' in response.json()


def test_v1_speech_accepts_cli_form_shape() -> None:
  """POST /v1/speech must accept the urlencoded form shape the CLI sends."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/v1/speech',
    data={'text': 'hello world', 'voice': 'af_heart', 'speed': '1.0'},
  )

  assert response.status_code != 422
  if response.status_code >= 400:
    assert 'error' in response.json()


def test_main_shim_exports_create_app() -> None:
  """Verify main.py is a valid shim that re-exports create_app."""
  from tts_gateway import main

  assert hasattr(main, 'create_app')
  assert callable(main.create_app)


def test_v1_speech_stream_accepts_json_body() -> None:
  """POST /v1/speech/stream must accept {text, voice, speed} JSON."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/v1/speech/stream',
    json={'text': 'hello world', 'voice': 'af_heart', 'speed': 1.0},
  )

  assert response.status_code != 422
  if response.status_code >= 400:
    assert 'error' in response.json()


def test_v1_speech_stream_accepts_pcm_format() -> None:
  """POST /v1/speech/stream with format='pcm' must not be rejected as invalid."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/v1/speech/stream',
    json={'text': 'hello world', 'format': 'pcm'},
  )

  assert response.status_code != 422
  if response.status_code >= 400:
    assert 'error' in response.json()


def test_v1_speech_stream_rejects_unknown_format() -> None:
  """POST /v1/speech/stream with an unsupported format must 422."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/v1/speech/stream',
    json={'text': 'hello world', 'format': 'ogg'},
  )

  assert response.status_code == 422
  assert 'error' in response.json()


def test_v1_speech_accepts_json_body() -> None:
  """POST /v1/speech must accept {text, voice, speed} JSON as well as form data."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/v1/speech',
    json={'text': 'hello world', 'voice': 'af_heart', 'speed': 1.0},
  )

  assert response.status_code != 422
  if response.status_code >= 400:
    assert 'error' in response.json()


def test_tts_stream_carries_deprecation_header() -> None:
  """Legacy POST /tts/stream must carry a Deprecation header."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/tts/stream',
    json={'text': 'hello world', 'voice': 'af_heart', 'speed': 1.0},
  )

  assert response.status_code != 422
  assert response.headers.get('Deprecation') == 'true'


def test_tts_stream_pcm_carries_deprecation_header() -> None:
  """Legacy POST /tts/stream/pcm must carry a Deprecation header."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post(
    '/tts/stream/pcm',
    json={'text': 'hello world', 'voice': 'af_heart', 'speed': 1.0},
  )

  assert response.status_code != 422
  assert response.headers.get('Deprecation') == 'true'


def test_legacy_tts_carries_deprecation_header() -> None:
  """Legacy POST /tts must carry a Deprecation header."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post('/tts', data={'text': 'hello world'})

  assert response.headers.get('Deprecation') == 'true'


def test_legacy_tts_sync_carries_deprecation_header() -> None:
  """Legacy POST /tts/sync must carry a Deprecation header."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post('/tts/sync', data={'text': 'hello world'})

  assert response.headers.get('Deprecation') == 'true'


def test_legacy_validation_errors_carry_deprecation_header() -> None:
  """FastAPI's own 422 responses on /tts/* must also carry the header."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post('/tts/stream', json={'voice': 'af_heart'})

  assert response.status_code == 422
  assert response.headers.get('Deprecation') == 'true'


def test_v1_speech_rejects_out_of_range_json_speed() -> None:
  """A JSON integer too large for float must yield 422, not 500."""
  app = create_app(_make_config(kokoro_enabled=False, pocket_enabled=False))
  client = TestClient(app)

  response = client.post('/v1/speech', json={'text': 'hello', 'speed': 10**400})

  assert response.status_code == 422
  assert 'speed' in response.json()['error']
