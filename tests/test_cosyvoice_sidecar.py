"""Tests for CosyVoice sidecar engine, config, CLI, and runtime wiring."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.conftest import MockEngine, MockStreamingEngine, _make_config
from tts_gateway.cli import _set_common_env
from tts_gateway.config import load_config
from tts_gateway.engines.base import EngineError
from tts_gateway.engines.cosyvoice_sidecar import CosyVoiceSidecarEngine
from tts_gateway.main import create_app
from tts_gateway.render import stream_pcm
from tts_gateway.runtime import JobRuntime
from tts_gateway.types import SynthesisSpec

_SIDEcar_HEADERS = {
  'content-type': 'audio/raw',
  'x-tts-sample-rate': '24000',
  'x-tts-channels': '1',
  'x-tts-sample-width': '2',
}


def _engine(**overrides: object) -> CosyVoiceSidecarEngine:
  config = _make_config(cosyvoice_enabled=True, **overrides)
  return CosyVoiceSidecarEngine(config)


def _pcm_bytes(samples: int, *, sample_width: int = 2) -> bytes:
  return b'\x01\x02' * (samples // sample_width)


class _FakeStreamResponse:
  def __init__(
    self,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    body_chunks: tuple[bytes, ...] = (),
    error_body: bytes = b'sidecar failed',
  ) -> None:
    self.status_code = status_code
    self.headers = headers or _SIDEcar_HEADERS
    self._body_chunks = body_chunks
    self._error_body = error_body

  async def aread(self) -> bytes:
    return self._error_body

  def aiter_bytes(self) -> AsyncIterator[bytes]:
    async def _iter() -> AsyncIterator[bytes]:
      for chunk in self._body_chunks:
        yield chunk

    return _iter()


class _FakeStreamContext:
  def __init__(self, response: _FakeStreamResponse) -> None:
    self._response = response

  async def __aenter__(self) -> _FakeStreamResponse:
    return self._response

  async def __aexit__(self, *_args: object) -> None:
    return None


class _FakeAsyncClient:
  def __init__(self, response: _FakeStreamResponse) -> None:
    self._response = response
    self.last_request: dict[str, Any] | None = None

  async def __aenter__(self) -> _FakeAsyncClient:
    return self

  async def __aexit__(self, *_args: object) -> None:
    return None

  def stream(
    self, method: str, url: str, *, json: dict[str, str] | None = None
  ) -> _FakeStreamContext:
    self.last_request = {'method': method, 'url': url, 'json': json}
    return _FakeStreamContext(self._response)


def _patch_async_client(
  monkeypatch: pytest.MonkeyPatch,
  response: _FakeStreamResponse,
) -> _FakeAsyncClient:
  client = _FakeAsyncClient(response)

  class _ClientFactory:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
      pass

    async def __aenter__(self) -> _FakeAsyncClient:
      return client

    async def __aexit__(self, *_args: object) -> None:
      return None

  monkeypatch.setattr(
    'tts_gateway.engines.cosyvoice_sidecar.httpx.AsyncClient',
    _ClientFactory,
  )
  return client


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _set_minimal_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
  defaults = {
    'TTS_PRIMARY_ENGINE': 'kokoro',
    'TTS_FALLBACK_ENGINE': 'none',
    'TTS_OUTPUT_FORMAT': 'wav',
    'KOKORO_TTS_ENABLED': 'true',
    'POCKET_TTS_ENABLED': 'false',
    'COSYVOICE_TTS_ENABLED': 'false',
    'TTS_DEVICE_MODE': 'auto',
    'TTS_MODELS_DIR': '/tmp/models',
  }
  defaults.update(overrides)
  for key, val in defaults.items():
    monkeypatch.setenv(key, val)


def test_load_config_cosyvoice_primary(monkeypatch: pytest.MonkeyPatch) -> None:
  _set_minimal_env(
    monkeypatch,
    TTS_PRIMARY_ENGINE='cosyvoice',
    COSYVOICE_TTS_ENABLED='true',
    TTS_COSYVOICE_BASE_URL='http://127.0.0.1:51000',
  )
  cfg = load_config()
  assert cfg.primary_engine == 'cosyvoice'
  assert cfg.cosyvoice_enabled is True
  assert cfg.cosyvoice_base_url == 'http://127.0.0.1:51000'
  assert cfg.cosyvoice_request_timeout_seconds == 360


def test_load_config_cosyvoice_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  _set_minimal_env(
    monkeypatch,
    TTS_FALLBACK_ENGINE='cosyvoice',
    COSYVOICE_TTS_ENABLED='true',
  )
  cfg = load_config()
  assert cfg.fallback_engine == 'cosyvoice'


def test_load_config_cosyvoice_timeout_override(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_minimal_env(
    monkeypatch,
    TTS_ENGINE_TIMEOUT_SECONDS='120',
    TTS_COSYVOICE_REQUEST_TIMEOUT_SECONDS='45',
  )
  cfg = load_config()
  assert cfg.engine_timeout_seconds == 120
  assert cfg.cosyvoice_request_timeout_seconds == 45


def test_load_config_rejects_unknown_engine(monkeypatch: pytest.MonkeyPatch) -> None:
  _set_minimal_env(monkeypatch, TTS_PRIMARY_ENGINE='unknown')
  with pytest.raises(ValueError, match='kokoro, pocket, cosyvoice'):
    load_config()


# ---------------------------------------------------------------------------
# CLI env wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  'provider,fallback,expected',
  [
    ('cosyvoice', 'none', 'true'),
    ('kokoro', 'cosyvoice', 'true'),
    ('kokoro', 'none', 'false'),
  ],
)
def test_cli_sets_cosyvoice_enabled_env(
  monkeypatch: pytest.MonkeyPatch,
  provider: str,
  fallback: str,
  expected: str,
) -> None:
  monkeypatch.delenv('COSYVOICE_TTS_ENABLED', raising=False)
  args = argparse.Namespace(
    provider=provider,
    fallback=fallback,
    device='auto',
    format='mp3',
    voice=None,
    models_dir=None,
    chunk_size=None,
  )
  _set_common_env(args)
  assert os.environ['COSYVOICE_TTS_ENABLED'] == expected


# ---------------------------------------------------------------------------
# Runtime chain and /health
# ---------------------------------------------------------------------------


def test_runtime_includes_cosyvoice_when_enabled(tmp_path) -> None:
  rt = JobRuntime(
    _make_config(
      data_dir=str(tmp_path / 'data'),
      primary_engine='cosyvoice',
      cosyvoice_enabled=True,
      kokoro_enabled=False,
    )
  )
  assert rt.engine_chain() == ['cosyvoice']
  assert len(rt.engines) == 1
  assert rt.engines[0].name == 'cosyvoice'
  rt.close()


def test_health_reports_cosyvoice_disabled(tmp_path) -> None:
  config = _make_config(
    data_dir=str(tmp_path / 'data'),
    cosyvoice_enabled=False,
    kokoro_enabled=False,
  )
  app = create_app(config)
  with TestClient(app) as client:
    body = client.get('/health').json()
  assert body['engines']['cosyvoice'] == {'mode': 'disabled'}


def test_health_reports_cosyvoice_sidecar(tmp_path) -> None:
  config = _make_config(
    data_dir=str(tmp_path / 'data'),
    primary_engine='cosyvoice',
    cosyvoice_enabled=True,
    cosyvoice_base_url='http://127.0.0.1:50000',
    kokoro_enabled=False,
  )
  app = create_app(config)
  with TestClient(app) as client:
    body = client.get('/health').json()
  assert body['engines']['cosyvoice'] == {
    'mode': 'sidecar',
    'enabled': True,
    'baseUrl': 'http://127.0.0.1:50000',
  }


# ---------------------------------------------------------------------------
# Sidecar stream parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_parses_headers_and_yields_frame_aligned_chunks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pcm = _pcm_bytes(8)
  client = _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(pcm[:3], pcm[3:])),
  )

  engine = _engine()
  chunks = [chunk async for chunk in engine.stream_synthesize('hello', voice='v1')]

  assert client.last_request == {
    'method': 'POST',
    'url': 'http://127.0.0.1:50000/v1/tts/stream',
    'json': {'text': 'hello', 'voice': 'v1'},
  }
  assert [len(chunk.pcm_bytes) for chunk in chunks] == [2, 6]
  assert all(chunk.sample_rate == 24_000 for chunk in chunks)
  assert all(chunk.channels == 1 for chunk in chunks)
  assert all(chunk.sample_width == 2 for chunk in chunks)
  assert b''.join(chunk.pcm_bytes for chunk in chunks) == pcm


@pytest.mark.asyncio
async def test_stream_buffers_split_frames_across_http_chunks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pcm = _pcm_bytes(6)
  _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(pcm[:1], pcm[1:3], pcm[3:])),
  )

  engine = _engine()
  chunks = [chunk async for chunk in engine.stream_synthesize('hello')]

  assert [len(chunk.pcm_bytes) for chunk in chunks] == [2, 4]
  assert b''.join(chunk.pcm_bytes for chunk in chunks) == pcm


@pytest.mark.asyncio
async def test_stream_accepts_title_case_audio_headers(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  headers = {
    'Content-Type': 'audio/raw; charset=binary',
    'X-TTS-Sample-Rate': '24000',
    'X-TTS-Channels': '1',
    'X-TTS-Sample-Width': '2',
  }
  _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(headers=headers, body_chunks=(b'\x00\x01',)),
  )
  engine = _engine()

  chunks = [chunk async for chunk in engine.stream_synthesize('hello')]

  assert len(chunks) == 1
  assert chunks[0].sample_rate == 24_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'headers,match',
  [
    ({'content-type': 'audio/wav'}, 'unsupported content type'),
    ({'content-type': 'audio/raw'}, 'x-tts-sample-rate'),
    (
      {
        'content-type': 'audio/raw',
        'x-tts-sample-rate': '24000',
        'x-tts-channels': '1',
      },
      'X-TTS-Sample-Width or X-TTS-Pcm-Format',
    ),
    (
      {
        'content-type': 'audio/raw',
        'x-tts-sample-rate': '24000',
        'x-tts-channels': '1',
        'x-tts-pcm-format': 'f32le',
      },
      'unsupported PCM format',
    ),
    (
      {
        'content-type': 'audio/raw',
        'x-tts-sample-rate': '24000',
        'x-tts-channels': '0',
        'x-tts-sample-width': '2',
      },
      'must be > 0',
    ),
  ],
)
async def test_stream_missing_or_unsupported_headers_raise(
  monkeypatch: pytest.MonkeyPatch,
  headers: dict[str, str],
  match: str,
) -> None:
  _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(headers=headers, body_chunks=(b'\x00\x01',)),
  )
  engine = _engine()

  with pytest.raises(EngineError, match=match):
    async for _ in engine.stream_synthesize('hello'):
      pass


@pytest.mark.asyncio
async def test_stream_non_2xx_raises_engine_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(status_code=503, error_body=b'overloaded'),
  )
  engine = _engine()

  with pytest.raises(EngineError, match='503'):
    async for _ in engine.stream_synthesize('hello'):
      pass


@pytest.mark.asyncio
async def test_stream_partial_trailing_frame_raises(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(b'\x00',)),
  )
  engine = _engine()

  with pytest.raises(EngineError, match='partial PCM frame'):
    async for _ in engine.stream_synthesize('hello'):
      pass


@pytest.mark.asyncio
async def test_stream_unreachable_sidecar_raises(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _FailingClientFactory:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
      pass

    async def __aenter__(self) -> None:
      raise httpx.ConnectError('connection refused')

    async def __aexit__(self, *_args: object) -> None:
      return None

  monkeypatch.setattr(
    'tts_gateway.engines.cosyvoice_sidecar.httpx.AsyncClient',
    _FailingClientFactory,
  )
  engine = _engine()

  with pytest.raises(EngineError, match='unreachable'):
    async for _ in engine.stream_synthesize('hello'):
      pass


@pytest.mark.asyncio
async def test_synthesize_merges_streamed_chunks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pcm = _pcm_bytes(4)
  _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(body_chunks=(pcm[:2], pcm[2:])),
  )
  engine = _engine()

  merged = await engine.synthesize('hello', voice='v1')

  assert merged.sample_rate == 24_000
  assert merged.channels == 1
  assert merged.sample_width == 2
  assert merged.pcm_bytes == pcm


@pytest.mark.asyncio
async def test_stream_accepts_pcm_format_header(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  headers = {
    'content-type': 'audio/raw',
    'x-tts-sample-rate': '16000',
    'x-tts-channels': '2',
    'x-tts-pcm-format': 's16le',
  }
  pcm = b'\x00\x01' * 4
  _patch_async_client(
    monkeypatch,
    _FakeStreamResponse(headers=headers, body_chunks=(pcm,)),
  )
  engine = _engine()

  chunks = [chunk async for chunk in engine.stream_synthesize('hello')]

  assert len(chunks) == 1
  assert chunks[0].sample_rate == 16_000
  assert chunks[0].channels == 2


# ---------------------------------------------------------------------------
# Renderer prefers CosyVoice native streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_renderer_prefers_cosyvoice_native_streaming() -> None:
  cosyvoice = MockStreamingEngine('cosyvoice')
  synth_only = MockEngine('kokoro')
  request = SynthesisSpec(text='hello world', voice='v', output_format='wav')

  first, rest = await stream_pcm(request, [cosyvoice, synth_only], engine_timeout=10)
  async for _ in rest:
    pass

  assert cosyvoice.stream_calls == [('hello world', 'v')]
  assert synth_only.calls == []
  assert first == cosyvoice.chunks[0]
