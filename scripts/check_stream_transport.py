#!/usr/bin/env python3
"""Fetch stream endpoints from a live gateway and validate payloads with ffmpeg."""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class StreamFetchResult:
  endpoint: str
  status: int
  content_type: str | None
  headers: dict[str, str]
  payload: bytes


@dataclass(frozen=True)
class DecodeCheck:
  endpoint: str
  ok: bool
  output_path: Path | None
  error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--base-url', default='http://127.0.0.1:45123')
  parser.add_argument('--text', default='Stream transport validation sample.')
  parser.add_argument('--output-dir', type=Path, default=Path('tmp/stream-transport'))
  parser.add_argument('--ffmpeg-path', default='ffmpeg')
  parser.add_argument(
    '--endpoint',
    action='append',
    dest='endpoints',
    help='Stream endpoint to validate (default: /tts/stream and /tts/stream/pcm)',
  )
  return parser.parse_args(argv)


def default_endpoints(args: argparse.Namespace) -> list[str]:
  return args.endpoints or ['/tts/stream', '/tts/stream/pcm']


def normalize_headers(raw_headers: http.client.HTTPMessage) -> dict[str, str]:
  return {key.lower(): value for key, value in raw_headers.items()}


def fetch_stream(base_url: str, endpoint: str, text: str) -> StreamFetchResult:
  parsed = urlparse(base_url)
  conn_cls = (
    http.client.HTTPSConnection
    if parsed.scheme == 'https'
    else http.client.HTTPConnection
  )
  conn = conn_cls(parsed.hostname or '', parsed.port, timeout=180)
  request_path = f'{parsed.path.rstrip("/")}{endpoint}' or endpoint
  body = json.dumps({'text': text}).encode()
  headers = {'Content-Type': 'application/json'}
  try:
    conn.request('POST', request_path, body=body, headers=headers)
    response = conn.getresponse()
    payload = response.read()
    return StreamFetchResult(
      endpoint=endpoint,
      status=response.status,
      content_type=response.getheader('Content-Type'),
      headers=normalize_headers(response.headers),
      payload=payload,
    )
  finally:
    conn.close()


def output_suffix(endpoint: str, content_type: str | None) -> str:
  if endpoint.endswith('/pcm') or content_type == 'audio/raw':
    return 'pcm'
  return 'mp3'


def pcm_decode_args(headers: dict[str, str], ffmpeg_path: str) -> list[str]:
  sample_rate = headers.get('x-tts-sample-rate', '24000')
  channels = headers.get('x-tts-channels', '1')
  pcm_format = headers.get('x-tts-pcm-format', 's16le')
  return [
    ffmpeg_path,
    '-hide_banner',
    '-loglevel',
    'error',
    '-f',
    pcm_format,
    '-ar',
    sample_rate,
    '-ac',
    channels,
    '-i',
    'pipe:0',
    '-f',
    'null',
    '-',
  ]


def mp3_decode_args(ffmpeg_path: str) -> list[str]:
  return [
    ffmpeg_path,
    '-hide_banner',
    '-loglevel',
    'error',
    '-i',
    'pipe:0',
    '-f',
    'null',
    '-',
  ]


def decode_command(
  endpoint: str,
  content_type: str | None,
  headers: dict[str, str],
  ffmpeg_path: str,
) -> list[str]:
  if endpoint.endswith('/pcm') or content_type == 'audio/raw':
    return pcm_decode_args(headers, ffmpeg_path)
  return mp3_decode_args(ffmpeg_path)


def decode_payload(
  *,
  endpoint: str,
  content_type: str | None,
  headers: dict[str, str],
  payload: bytes,
  ffmpeg_path: str,
) -> None:
  command = decode_command(endpoint, content_type, headers, ffmpeg_path)
  process = subprocess.run(
    command,
    input=payload,
    capture_output=True,
    check=False,
  )
  if process.returncode != 0:
    stderr = process.stderr.decode(errors='replace').strip() or 'ffmpeg decode failed'
    raise RuntimeError(stderr)


def validate_fetch(
  fetch: StreamFetchResult,
  *,
  output_dir: Path,
  ffmpeg_path: str,
) -> DecodeCheck:
  if fetch.status != 200:
    return DecodeCheck(
      endpoint=fetch.endpoint,
      ok=False,
      output_path=None,
      error=f'HTTP {fetch.status}',
    )

  output_dir.mkdir(parents=True, exist_ok=True)
  suffix = output_suffix(fetch.endpoint, fetch.content_type)
  output_path = output_dir / f'{suffix}{fetch.endpoint.replace("/", "_")}.bin'
  output_path.write_bytes(fetch.payload)

  try:
    decode_payload(
      endpoint=fetch.endpoint,
      content_type=fetch.content_type,
      headers=fetch.headers,
      payload=fetch.payload,
      ffmpeg_path=ffmpeg_path,
    )
  except RuntimeError as exc:
    return DecodeCheck(
      endpoint=fetch.endpoint,
      ok=False,
      output_path=output_path,
      error=str(exc),
    )

  return DecodeCheck(endpoint=fetch.endpoint, ok=True, output_path=output_path)


def run_checks(args: argparse.Namespace) -> list[DecodeCheck]:
  results: list[DecodeCheck] = []
  for endpoint in default_endpoints(args):
    fetch = fetch_stream(args.base_url, endpoint, args.text)
    results.append(
      validate_fetch(
        fetch,
        output_dir=args.output_dir,
        ffmpeg_path=args.ffmpeg_path,
      )
    )
  return results


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  checks = run_checks(args)
  failed = [check for check in checks if not check.ok]
  for check in checks:
    if check.ok:
      print(f'OK  {check.endpoint} -> {check.output_path}')
    else:
      print(f'FAIL {check.endpoint}: {check.error}', file=sys.stderr)
  return 1 if failed else 0


if __name__ == '__main__':
  raise SystemExit(main())
