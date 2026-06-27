"""Tests for scripts/check_stream_transport.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  'check_stream_transport',
  ROOT / 'scripts' / 'check_stream_transport.py',
)
assert SPEC is not None and SPEC.loader is not None
check = importlib.util.module_from_spec(SPEC)
sys.modules['check_stream_transport'] = check
SPEC.loader.exec_module(check)


def test_default_endpoints() -> None:
  args = check.parse_args(['--base-url', 'http://127.0.0.1:8000'])
  assert check.default_endpoints(args) == ['/tts/stream', '/tts/stream/pcm']


def test_custom_endpoints() -> None:
  args = check.parse_args(['--endpoint', '/tts/stream/pcm'])
  assert check.default_endpoints(args) == ['/tts/stream/pcm']


def test_pcm_decode_args_from_headers() -> None:
  command = check.pcm_decode_args(
    {
      'x-tts-sample-rate': '24000',
      'x-tts-channels': '1',
      'x-tts-pcm-format': 's16le',
    },
    'ffmpeg',
  )
  assert command == [
    'ffmpeg',
    '-hide_banner',
    '-loglevel',
    'error',
    '-f',
    's16le',
    '-ar',
    '24000',
    '-ac',
    '1',
    '-i',
    'pipe:0',
    '-f',
    'null',
    '-',
  ]


def test_mp3_decode_args() -> None:
  assert check.mp3_decode_args('ffmpeg') == [
    'ffmpeg',
    '-hide_banner',
    '-loglevel',
    'error',
    '-i',
    'pipe:0',
    '-f',
    'null',
    '-',
  ]


def test_decode_command_selects_pcm_for_raw_content_type() -> None:
  command = check.decode_command(
    '/tts/stream/pcm',
    'audio/raw',
    {'x-tts-pcm-format': 's16le', 'x-tts-sample-rate': '22050', 'x-tts-channels': '2'},
    'ffmpeg',
  )
  assert '-f' in command
  assert command[command.index('-f') + 1] == 's16le'
  assert command[command.index('-ar') + 1] == '22050'
  assert command[command.index('-ac') + 1] == '2'


def test_decode_payload_passes_binary_audio_to_ffmpeg() -> None:
  with patch('check_stream_transport.subprocess.run') as run:
    run.return_value.returncode = 0
    run.return_value.stderr = b''

    check.decode_payload(
      endpoint='/tts/stream/pcm',
      content_type='audio/raw',
      headers={
        'x-tts-sample-rate': '24000',
        'x-tts-channels': '1',
        'x-tts-pcm-format': 's16le',
      },
      payload=b'\x00\x00',
      ffmpeg_path='ffmpeg',
    )

  kwargs = run.call_args.kwargs
  assert kwargs['input'] == b'\x00\x00'
  assert kwargs['capture_output'] is True
  assert 'text' not in kwargs


def test_validate_fetch_writes_payload_and_decodes(tmp_path) -> None:
  payload = b'\x00\x00' * 100
  fetch = check.StreamFetchResult(
    endpoint='/tts/stream/pcm',
    status=200,
    content_type='audio/raw',
    headers={
      'x-tts-sample-rate': '24000',
      'x-tts-channels': '1',
      'x-tts-pcm-format': 's16le',
    },
    payload=payload,
  )

  with patch('check_stream_transport.decode_payload') as decode_mock:
    result = check.validate_fetch(
      fetch,
      output_dir=tmp_path,
      ffmpeg_path='ffmpeg',
    )

  assert result.ok is True
  assert result.output_path is not None
  assert result.output_path.read_bytes() == payload
  decode_mock.assert_called_once()


def test_validate_fetch_reports_http_error(tmp_path) -> None:
  fetch = check.StreamFetchResult(
    endpoint='/tts/stream',
    status=503,
    content_type='application/json',
    headers={},
    payload=b'{"error":"all engines in chain are unavailable"}',
  )
  result = check.validate_fetch(fetch, output_dir=tmp_path, ffmpeg_path='ffmpeg')
  assert result.ok is False
  assert result.error == 'HTTP 503'


def test_validate_fetch_reports_decode_error(tmp_path) -> None:
  fetch = check.StreamFetchResult(
    endpoint='/tts/stream',
    status=200,
    content_type='audio/mpeg',
    headers={},
    payload=b'not-mp3',
  )

  with patch(
    'check_stream_transport.decode_payload',
    side_effect=RuntimeError('invalid data'),
  ):
    result = check.validate_fetch(fetch, output_dir=tmp_path, ffmpeg_path='ffmpeg')

  assert result.ok is False
  assert result.error == 'invalid data'


def test_main_returns_nonzero_on_failure() -> None:
  with patch(
    'check_stream_transport.run_checks',
    return_value=[
      check.DecodeCheck('/tts/stream', ok=False, output_path=None, error='x')
    ],
  ):
    assert check.main([]) == 1


def test_main_returns_zero_on_success() -> None:
  with patch(
    'check_stream_transport.run_checks',
    return_value=[check.DecodeCheck('/tts/stream', ok=True, output_path=Path('x.bin'))],
  ):
    assert check.main([]) == 0
