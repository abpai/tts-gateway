"""Tests for scripts/cosyvoice_official_bridge.py."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  'cosyvoice_official_bridge',
  ROOT / 'scripts' / 'cosyvoice_official_bridge.py',
)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
sys.modules['cosyvoice_official_bridge'] = bridge
SPEC.loader.exec_module(bridge)


def _settings(**overrides: object) -> bridge.BridgeSettings:
  defaults: dict[str, object] = {
    'mode': 'sft',
    'upstream_base_url': 'http://127.0.0.1:50001',
    'upstream_endpoint': '/inference_sft',
    'default_voice': 'default-spk',
    'prompt_text': None,
    'prompt_wav_path': None,
    'instruct_text': None,
    'sample_rate': 22050,
    'channels': 1,
    'pcm_format': 's16le',
    'sample_width': 2,
    'request_timeout': 30.0,
  }
  defaults.update(overrides)
  return bridge.BridgeSettings(**defaults)


class _FakeStreamResponse:
  def __init__(
    self,
    *,
    status_code: int = 200,
    body_chunks: tuple[bytes, ...] = (),
    error_body: bytes = b'upstream failed',
  ) -> None:
    self.status_code = status_code
    self._body_chunks = body_chunks
    self._error_body = error_body

  async def aread(self) -> bytes:
    return self._error_body

  def aiter_bytes(self) -> AsyncIterator[bytes]:
    async def _iter() -> AsyncIterator[bytes]:
      for chunk in self._body_chunks:
        yield chunk

    return _iter()

  async def aclose(self) -> None:
    return None


class _FakeAsyncClient:
  def __init__(self, response: _FakeStreamResponse) -> None:
    self._response = response
    self.last_request: dict[str, Any] | None = None
    self.closed = False

  def build_request(
    self,
    method: str,
    url: str,
    *,
    data: dict[str, str] | None = None,
    files: dict[str, tuple[str, object, str]] | None = None,
  ) -> dict[str, Any]:
    self.last_request = {
      'method': method,
      'url': url,
      'data': data,
      'files': files,
    }
    return self.last_request

  async def send(
    self, _request: dict[str, Any], *, stream: bool = False
  ) -> _FakeStreamResponse:
    return self._response

  async def aclose(self) -> None:
    self.closed = True


def _patch_upstream_client(
  monkeypatch: pytest.MonkeyPatch,
  response: _FakeStreamResponse,
) -> _FakeAsyncClient:
  shared = _FakeAsyncClient(response)

  class _ClientFactory:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
      self._client = shared

    def build_request(
      self,
      method: str,
      url: str,
      *,
      data: dict[str, str] | None = None,
      files: dict[str, tuple[str, object, str]] | None = None,
    ) -> dict[str, Any]:
      return self._client.build_request(method, url, data=data, files=files)

    async def send(
      self, request: dict[str, Any], *, stream: bool = False
    ) -> _FakeStreamResponse:
      return await self._client.send(request, stream=stream)

    async def aclose(self) -> None:
      await self._client.aclose()

  monkeypatch.setattr(bridge.httpx, 'AsyncClient', _ClientFactory)
  return shared


def _patch_upstream_probe(
  monkeypatch: pytest.MonkeyPatch,
  *,
  reachable: bool,
) -> None:
  async def _probe(_settings: bridge.BridgeSettings) -> bool:
    return reachable

  monkeypatch.setattr(bridge, '_probe_upstream_reachable', _probe)


@pytest.fixture
def prompt_wav(tmp_path: Path) -> Path:
  path = tmp_path / 'prompt.wav'
  path.write_bytes(b'RIFFfake-wav')
  return path


def test_parse_args_defaults() -> None:
  args = bridge.parse_args(['--default-voice', 'spk-1'])
  assert args.mode == 'sft'
  assert args.upstream_base_url == 'http://127.0.0.1:50001'
  assert args.upstream_endpoint is None
  assert args.host == '127.0.0.1'
  assert args.port == 50000
  assert args.default_voice == 'spk-1'
  assert args.prompt_text is None
  assert args.prompt_wav is None
  assert args.instruct_text is None
  assert args.sample_rate == 22050
  assert args.request_timeout == 360.0
  assert args.debug is False


def test_settings_sft_defaults_are_compatible() -> None:
  args = bridge.parse_args(['--default-voice', 'spk-1'])
  settings = bridge.settings_from_args(args)
  assert settings.mode == 'sft'
  assert settings.upstream_endpoint == '/inference_sft'
  assert settings.default_voice == 'spk-1'


def test_english_narration_sets_default_instruct_text() -> None:
  args = bridge.parse_args(
    [
      '--mode',
      'instruct',
      '--default-voice',
      'spk-1',
      '--english-narration',
    ]
  )
  settings = bridge.settings_from_args(args)
  assert settings.instruct_text == bridge.DEFAULT_ENGLISH_NARRATION_INSTRUCT


def test_english_narration_still_requires_zero_shot_prompt_text(
  prompt_wav: Path,
) -> None:
  args = bridge.parse_args(
    [
      '--mode',
      'zero-shot',
      '--english-narration',
      '--prompt-wav',
      str(prompt_wav),
    ]
  )
  with pytest.raises(SystemExit, match='--prompt-text must not be empty'):
    bridge.settings_from_args(args)


def test_english_narration_preserves_zero_shot_prompt_transcript(
  prompt_wav: Path,
) -> None:
  args = bridge.parse_args(
    [
      '--mode',
      'zero-shot',
      '--english-narration',
      '--prompt-text',
      'This is the transcript of the English reference.',
      '--prompt-wav',
      str(prompt_wav),
    ]
  )
  settings = bridge.settings_from_args(args)
  assert settings.prompt_text == 'This is the transcript of the English reference.'


@pytest.mark.parametrize(
  ('mode', 'endpoint'),
  [
    ('sft', '/inference_sft'),
    ('zero-shot', '/inference_zero_shot'),
    ('cross-lingual', '/inference_cross_lingual'),
    ('instruct', '/inference_instruct'),
    ('instruct2', '/inference_instruct2'),
  ],
)
def test_settings_mode_derived_endpoint(
  mode: str,
  endpoint: str,
  prompt_wav: Path,
) -> None:
  argv = ['--mode', mode]
  if mode == 'sft':
    argv.extend(['--default-voice', 'spk-1'])
  elif mode == 'zero-shot':
    argv.extend(['--prompt-text', 'prompt transcript', '--prompt-wav', str(prompt_wav)])
  elif mode == 'cross-lingual':
    argv.extend(['--prompt-wav', str(prompt_wav)])
  elif mode == 'instruct':
    argv.extend(['--default-voice', 'spk-1', '--instruct-text', 'speak slowly'])
  elif mode == 'instruct2':
    argv.extend(['--instruct-text', 'speak slowly', '--prompt-wav', str(prompt_wav)])

  settings = bridge.settings_from_args(bridge.parse_args(argv))
  assert settings.upstream_endpoint == endpoint


def test_settings_normalizes_upstream_endpoint_override() -> None:
  args = bridge.parse_args(
    [
      '--default-voice',
      'spk-1',
      '--upstream-endpoint',
      'inference_sft',
    ]
  )
  settings = bridge.settings_from_args(args)
  assert settings.upstream_endpoint == '/inference_sft'


@pytest.mark.parametrize(
  'args,match',
  [
    (['--default-voice', 'spk-1', '--sample-rate', '0'], 'sample-rate'),
    (['--default-voice', 'spk-1', '--request-timeout', '0'], 'request-timeout'),
    (
      ['--default-voice', 'spk-1', '--upstream-endpoint', ''],
      'upstream-endpoint',
    ),
    (['--mode', 'sft'], 'default-voice'),
    (
      ['--mode', 'zero-shot', '--prompt-text', 'hello'],
      'prompt-wav',
    ),
    (['--mode', 'cross-lingual'], 'prompt-wav'),
    (
      ['--mode', 'instruct', '--default-voice', 'spk-1'],
      'instruct-text',
    ),
    (
      ['--mode', 'instruct2', '--instruct-text', 'slow'],
      'prompt-wav',
    ),
  ],
)
def test_settings_rejects_invalid_values(args: list[str], match: str) -> None:
  parsed = bridge.parse_args(args)
  with pytest.raises(SystemExit, match=match):
    bridge.settings_from_args(parsed)


def test_settings_rejects_missing_prompt_wav_file() -> None:
  args = bridge.parse_args(
    [
      '--mode',
      'cross-lingual',
      '--prompt-wav',
      '/tmp/does-not-exist-prompt.wav',
    ]
  )
  with pytest.raises(SystemExit, match='prompt-wav'):
    bridge.settings_from_args(args)


def test_health_response(monkeypatch: pytest.MonkeyPatch) -> None:
  _patch_upstream_probe(monkeypatch, reachable=True)
  app = bridge.create_app(_settings(default_voice='spk-main'))
  with TestClient(app) as client:
    body = client.get('/health').json()

  assert body == {
    'status': 'ok',
    'bridge': 'cosyvoice-official',
    'mode': 'sft',
    'upstreamBaseUrl': 'http://127.0.0.1:50001',
    'upstreamEndpoint': '/inference_sft',
    'sampleRate': 22050,
    'channels': 1,
    'pcmFormat': 's16le',
    'defaultVoice': 'spk-main',
    'promptTextConfigured': False,
    'promptWavConfigured': False,
    'instructTextConfigured': False,
    'upstreamReachable': True,
  }


def test_health_includes_mode_config_flags(
  monkeypatch: pytest.MonkeyPatch,
  prompt_wav: Path,
) -> None:
  _patch_upstream_probe(monkeypatch, reachable=False)
  app = bridge.create_app(
    _settings(
      mode='zero-shot',
      upstream_endpoint='/inference_zero_shot',
      default_voice=None,
      prompt_text='reference transcript',
      prompt_wav_path=prompt_wav,
    )
  )
  with TestClient(app) as client:
    body = client.get('/health').json()

  assert body['mode'] == 'zero-shot'
  assert body['promptTextConfigured'] is True
  assert body['promptWavConfigured'] is True
  assert body['instructTextConfigured'] is False
  assert body['promptWavBasename'] == 'prompt.wav'
  assert '/Users/' not in str(body.get('promptWavBasename', ''))


def test_stream_forwards_form_data_and_returns_audio_headers(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pcm = b'\x01\x02\x03\x04'
  upstream = _patch_upstream_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(pcm[:2], pcm[2:])),
  )
  app = bridge.create_app(_settings(default_voice='default-spk'))
  with TestClient(app) as client:
    response = client.post('/v1/tts/stream', json={'text': 'hello world'})

  assert response.status_code == 200
  assert response.headers['content-type'] == 'audio/raw'
  assert response.headers['x-tts-sample-rate'] == '22050'
  assert response.headers['x-tts-channels'] == '1'
  assert response.headers['x-tts-pcm-format'] == 's16le'
  assert response.headers['x-tts-sample-width'] == '2'
  assert response.headers['x-tts-upstream-endpoint'] == '/inference_sft'
  assert response.content == pcm
  assert upstream.last_request == {
    'method': 'POST',
    'url': 'http://127.0.0.1:50001/inference_sft',
    'data': {'tts_text': 'hello world', 'spk_id': 'default-spk'},
    'files': None,
  }
  assert upstream.closed is True


def test_zero_shot_sends_prompt_text_and_wav_multipart(
  monkeypatch: pytest.MonkeyPatch,
  prompt_wav: Path,
) -> None:
  upstream = _patch_upstream_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(b'\x00\x01',)),
  )
  app = bridge.create_app(
    _settings(
      mode='zero-shot',
      upstream_endpoint='/inference_zero_shot',
      default_voice=None,
      prompt_text='reference transcript',
      prompt_wav_path=prompt_wav,
    )
  )
  with TestClient(app) as client:
    client.post('/v1/tts/stream', json={'text': 'target sentence'})

  assert upstream.last_request is not None
  assert upstream.last_request['url'].endswith('/inference_zero_shot')
  assert upstream.last_request['data'] == {
    'tts_text': 'target sentence',
    'prompt_text': 'reference transcript',
  }
  files = upstream.last_request['files']
  assert files is not None
  filename, content, content_type = files['prompt_wav']
  assert filename == 'prompt.wav'
  assert content_type == 'audio/wav'
  assert content == b'RIFFfake-wav'


def test_instruct_sends_spk_id_and_instruct_text(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  upstream = _patch_upstream_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(b'\x00\x01',)),
  )
  app = bridge.create_app(
    _settings(
      mode='instruct',
      upstream_endpoint='/inference_instruct',
      instruct_text='speak slowly',
    )
  )
  with TestClient(app) as client:
    client.post('/v1/tts/stream', json={'text': 'hello', 'voice': 'custom-spk'})

  assert upstream.last_request is not None
  assert upstream.last_request['data'] == {
    'tts_text': 'hello',
    'spk_id': 'custom-spk',
    'instruct_text': 'speak slowly',
  }
  assert upstream.last_request['files'] is None


def test_stream_request_voice_overrides_default(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  upstream = _patch_upstream_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(b'\x00\x01',)),
  )
  app = bridge.create_app(_settings(default_voice='default-spk'))
  with TestClient(app) as client:
    client.post('/v1/tts/stream', json={'text': 'hello', 'voice': 'custom-spk'})

  assert upstream.last_request is not None
  assert upstream.last_request['data']['spk_id'] == 'custom-spk'


def test_stream_blank_voice_uses_default(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  upstream = _patch_upstream_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(b'\x00\x01',)),
  )
  app = bridge.create_app(_settings(default_voice='default-spk'))
  with TestClient(app) as client:
    client.post('/v1/tts/stream', json={'text': 'hello', 'voice': '   '})

  assert upstream.last_request is not None
  assert upstream.last_request['data']['spk_id'] == 'default-spk'


def test_stream_empty_text_returns_422() -> None:
  app = bridge.create_app(_settings())
  with TestClient(app) as client:
    response = client.post('/v1/tts/stream', json={'text': '   '})

  assert response.status_code == 422


def test_stream_non_2xx_upstream_returns_non_2xx(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _patch_upstream_client(
    monkeypatch,
    _FakeStreamResponse(status_code=503, error_body=b'overloaded'),
  )
  app = bridge.create_app(_settings())
  with TestClient(app) as client:
    response = client.post('/v1/tts/stream', json={'text': 'hello'})

  assert response.status_code == 503
  assert response.json()['detail'] == 'overloaded'


def test_stream_upstream_request_error_returns_non_2xx(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _FailingClient:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
      pass

    def build_request(self, *_args: object, **_kwargs: object) -> dict[str, str]:
      return {}

    async def send(self, *_args: object, **_kwargs: object) -> None:
      raise httpx.ConnectError('connection refused')

    async def aclose(self) -> None:
      return None

  monkeypatch.setattr(bridge.httpx, 'AsyncClient', _FailingClient)
  app = bridge.create_app(_settings())
  with TestClient(app) as client:
    response = client.post('/v1/tts/stream', json={'text': 'hello'})

  assert response.status_code == 502
  assert 'connection refused' in response.json()['detail']
