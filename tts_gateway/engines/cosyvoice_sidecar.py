from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping

import httpx

from tts_gateway.audio import merge_chunks
from tts_gateway.config import GatewayConfig
from tts_gateway.engines.base import AudioChunk, EngineError, TtsEngine

_PCM_FORMAT_TO_WIDTH = {
  'u8': 1,
  's16le': 2,
  's32le': 4,
}

_SUPPORTED_CONTENT_TYPE = 'audio/raw'


class CosyVoiceSidecarEngine(TtsEngine):
  """HTTP sidecar-backed CosyVoice engine."""

  name = 'cosyvoice'

  def __init__(self, config: GatewayConfig) -> None:
    self._enabled = config.cosyvoice_enabled
    self._base_url = config.cosyvoice_base_url.rstrip('/')
    self._timeout = config.cosyvoice_request_timeout_seconds

  def health_status(self) -> dict[str, object]:
    if not self._enabled:
      return {'mode': 'disabled'}
    return {
      'mode': 'sidecar',
      'enabled': True,
      'baseUrl': self._base_url,
    }

  async def stream_synthesize(
    self, text: str, *, voice: str | None = None
  ) -> AsyncGenerator[AudioChunk, None]:
    if not self._enabled:
      raise EngineError('cosyvoice engine is disabled')

    payload: dict[str, str] = {'text': text}
    if voice:
      payload['voice'] = voice

    try:
      async with httpx.AsyncClient(timeout=self._timeout) as client:
        async with client.stream(
          'POST',
          f'{self._base_url}/v1/tts/stream',
          json=payload,
        ) as response:
          if response.status_code < 200 or response.status_code >= 300:
            detail = (await response.aread()).decode('utf-8', errors='replace')
            raise EngineError(
              f'cosyvoice sidecar returned {response.status_code}: {detail}'
            )

          sample_rate, channels, sample_width = _parse_audio_headers(response.headers)
          frame_bytes = sample_width * channels
          pending = bytearray()

          async for data in response.aiter_bytes():
            pending.extend(data)
            complete_bytes = len(pending) - (len(pending) % frame_bytes)
            if complete_bytes:
              audio_payload = bytes(pending[:complete_bytes])
              del pending[:complete_bytes]
              yield AudioChunk(
                pcm_bytes=audio_payload,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
              )

          if pending:
            raise EngineError('cosyvoice sidecar stream ended with partial PCM frame')
    except httpx.RequestError as exc:
      raise EngineError(f'cosyvoice sidecar unreachable: {exc}') from exc

  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    chunks: list[AudioChunk] = []
    async for chunk in self.stream_synthesize(text, voice=voice):
      chunks.append(chunk)
    if not chunks:
      raise EngineError('cosyvoice sidecar returned no audio')
    return merge_chunks(chunks)


def _parse_audio_headers(headers: Mapping[str, str]) -> tuple[int, int, int]:
  normalized = {key.lower(): value for key, value in headers.items()}
  content_type = normalized.get('content-type', '').split(';', 1)[0].strip().lower()
  if content_type != _SUPPORTED_CONTENT_TYPE:
    raise EngineError(
      f'cosyvoice sidecar returned unsupported content type: {content_type or "(missing)"}'
    )

  sample_rate = _required_positive_int_header(normalized, 'x-tts-sample-rate')
  channels = _required_positive_int_header(normalized, 'x-tts-channels')
  sample_width = _resolve_sample_width(normalized)

  if sample_width not in _PCM_FORMAT_TO_WIDTH.values():
    raise EngineError(
      f'cosyvoice sidecar returned unsupported sample width: {sample_width}'
    )

  return sample_rate, channels, sample_width


def _required_positive_int_header(headers: Mapping[str, str], name: str) -> int:
  raw = headers.get(name, '').strip()
  if not raw:
    raise EngineError(f'cosyvoice sidecar response missing required header: {name}')
  try:
    parsed = int(raw)
  except ValueError as exc:
    raise EngineError(
      f'cosyvoice sidecar header {name} must be an integer, received: {raw}'
    ) from exc
  if parsed <= 0:
    raise EngineError(f'cosyvoice sidecar header {name} must be > 0, received: {raw}')
  return parsed


def _resolve_sample_width(headers: Mapping[str, str]) -> int:
  width_raw = headers.get('x-tts-sample-width', '').strip()
  if width_raw:
    try:
      return int(width_raw)
    except ValueError as exc:
      raise EngineError(
        'cosyvoice sidecar header X-TTS-Sample-Width must be an integer, '
        f'received: {width_raw}'
      ) from exc

  pcm_format = headers.get('x-tts-pcm-format', '').strip().lower()
  if not pcm_format:
    raise EngineError(
      'cosyvoice sidecar response missing X-TTS-Sample-Width or X-TTS-Pcm-Format'
    )

  sample_width = _PCM_FORMAT_TO_WIDTH.get(pcm_format)
  if sample_width is None:
    raise EngineError(
      f'cosyvoice sidecar returned unsupported PCM format: {pcm_format}'
    )
  return sample_width
