#!/usr/bin/env python3
"""Manual API test script for tts-gateway.

Usage:
  # Start the server first:
  tts serve --provider kokoro --format mp3

  # Then run this script:
  python scripts/test_api.py

  # With a long text file:
  python scripts/test_api.py --long-text /tmp/long-text.md

  # Custom base URL:
  python scripts/test_api.py --base-url http://localhost:9000

Each test prints PASS/FAIL and saves audio to /tmp/tts-test/.
Step through interactively with --step to pause between tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OUTPUT_DIR = Path('/tmp/tts-test')
BASE_URL = 'http://localhost:8000'


def _request(
  method: str,
  path: str,
  *,
  data: bytes | None = None,
  headers: dict[str, str] | None = None,
  base_url: str = BASE_URL,
) -> tuple[int, bytes, dict[str, str]]:
  url = f'{base_url}{path}'
  req = Request(url, data=data, method=method)
  if headers:
    for k, v in headers.items():
      req.add_header(k, v)
  try:
    resp = urlopen(req, timeout=300)
    return resp.status, resp.read(), dict(resp.headers)
  except HTTPError as e:
    return e.code, e.read(), dict(e.headers)


def _post_form(path: str, fields: dict, **kwargs) -> tuple[int, bytes, dict]:
  body = '&'.join(f'{k}={v}' for k, v in fields.items()).encode()
  return _request(
    'POST',
    path,
    data=body,
    headers={
      'Content-Type': 'application/x-www-form-urlencoded',
      **kwargs.get('headers', {}),
    },
    **{k: v for k, v in kwargs.items() if k != 'headers'},
  )


def _post_json(path: str, obj: dict, **kwargs) -> tuple[int, bytes, dict]:
  body = json.dumps(obj).encode()
  return _request(
    'POST',
    path,
    data=body,
    headers={'Content-Type': 'application/json', **kwargs.get('headers', {})},
    **{k: v for k, v in kwargs.items() if k != 'headers'},
  )


def _get(path: str, **kwargs) -> tuple[int, bytes, dict]:
  return _request('GET', path, **kwargs)


class TestRunner:
  def __init__(self, base_url: str, step: bool = False):
    self.base_url = base_url
    self.step = step
    self.passed = 0
    self.failed = 0

  def _check(self, name: str, condition: bool, detail: str = ''):
    if condition:
      print(f'  PASS  {name}')
      self.passed += 1
    else:
      print(f'  FAIL  {name}{f" — {detail}" if detail else ""}')
      self.failed += 1

  def _pause(self):
    if self.step:
      input('  Press Enter to continue...')

  def test_health(self):
    print('\n--- 1. Health Check ---')
    status, body, _ = _get('/health', base_url=self.base_url)
    self._check('GET /health returns 200', status == 200)
    data = json.loads(body)
    self._check('Response has ok=true', data.get('ok') is True)
    self._check('Engine chain present', len(data.get('engineChain', [])) > 0)
    print(f'  Engine chain: {data.get("engineChain")}')
    print(f'  Output format: {data.get("outputFormat")}')
    self._pause()

  def test_warmup(self):
    print('\n--- 2. Warmup ---')
    try:
      status, body, _ = _request('POST', '/warmup', base_url=self.base_url)
    except Exception as e:
      self._check('POST /warmup returns 200', False, f'request failed: {e}')
      self._pause()
      return
    self._check('POST /warmup returns 200', status == 200)
    try:
      data = json.loads(body)
      print(f'  Engines: {json.dumps(data.get("engines", {}), indent=2)}')
    except json.JSONDecodeError:
      print(f'  Response not JSON: {body[:200]}')
    self._pause()

  def test_legacy_tts(self):
    print('\n--- 3. Legacy Buffered TTS (POST /tts, default Accept) ---')
    start = time.perf_counter()
    status, body, headers = _post_form(
      '/tts',
      {'text': 'Hello world. This is the legacy buffered endpoint.'},
      base_url=self.base_url,
    )
    elapsed = time.perf_counter() - start
    ct = headers.get('Content-Type', headers.get('content-type', ''))
    self._check('Returns 200', status == 200)
    self._check('Returns audio content-type', 'audio/' in ct, f'got {ct}')
    self._check('Returns non-empty body', len(body) > 100, f'{len(body)} bytes')
    out = OUTPUT_DIR / f'legacy.{"mp3" if "mpeg" in ct else "wav"}'
    out.write_bytes(body)
    print(f'  Saved: {out} ({len(body):,} bytes, {elapsed:.2f}s)')
    self._pause()

  def test_legacy_tts_with_voice(self):
    print('\n--- 4. Legacy TTS with Voice ---')
    status, body, headers = _post_form(
      '/tts',
      {'text': 'Testing voice selection.', 'voice': 'af_heart'},
      base_url=self.base_url,
    )
    ct = headers.get('Content-Type', headers.get('content-type', ''))
    self._check('Returns 200', status == 200)
    self._check('Returns audio', len(body) > 100)
    out = OUTPUT_DIR / f'voice.{"mp3" if "mpeg" in ct else "wav"}'
    out.write_bytes(body)
    print(f'  Saved: {out} ({len(body):,} bytes)')
    self._pause()

  def test_sync_endpoint(self):
    print('\n--- 5. Explicit Sync Endpoint (POST /tts/sync) ---')
    status, body, headers = _post_form(
      '/tts/sync',
      {'text': 'This is the explicit sync endpoint.'},
      base_url=self.base_url,
    )
    ct = headers.get('Content-Type', headers.get('content-type', ''))
    self._check('Returns 200', status == 200)
    self._check('Returns audio', 'audio/' in ct)
    out = OUTPUT_DIR / f'sync.{"mp3" if "mpeg" in ct else "wav"}'
    out.write_bytes(body)
    print(f'  Saved: {out} ({len(body):,} bytes)')
    self._pause()

  def test_streaming(self):
    print('\n--- 6. Streaming TTS (POST /tts/stream) ---')
    start = time.perf_counter()
    status, body, headers = _post_json(
      '/tts/stream',
      {
        'text': 'First sentence for streaming. Second sentence arrives next. Third sentence wraps it up.'
      },
      base_url=self.base_url,
    )
    elapsed = time.perf_counter() - start
    ct = headers.get('Content-Type', headers.get('content-type', ''))
    self._check('Returns 200', status == 200)
    self._check('Content-Type is audio/mpeg', 'audio/mpeg' in ct, f'got {ct}')
    self._check('Returns non-empty body', len(body) > 100, f'{len(body)} bytes')
    out = OUTPUT_DIR / 'stream.mp3'
    out.write_bytes(body)
    print(f'  Saved: {out} ({len(body):,} bytes, {elapsed:.2f}s)')
    self._pause()

  def test_job_submit(self):
    print('\n--- 7. Job Submit (POST /tts, Accept: application/json) ---')
    # Use unique text so we always get a fresh job (not a cache hit)
    unique_text = (
      f'Job test run {int(time.time())}. This is a job-based synthesis request.'
    )
    status, body, _ = _post_form(
      '/tts',
      {'text': unique_text},
      headers={'Accept': 'application/json'},
      base_url=self.base_url,
    )
    data = json.loads(body)
    self._check('Returns 202 (queued)', status == 202)
    self._check('Has job key', bool(data.get('key')))
    self._check('Status is queued', data.get('status') == 'queued')
    print(f'  Job key: {data.get("key", "?")[:24]}...')
    return data.get('key')

  def test_job_poll_and_download(self, job_key: str):
    print('\n--- 8. Job Poll + Download ---')
    if not job_key:
      print('  SKIP  No job key from previous test')
      return

    # Poll until ready or timeout
    deadline = time.time() + 60
    status_val = 'queued'
    while time.time() < deadline:
      status, body, _ = _get(f'/tts/{job_key}', base_url=self.base_url)
      data = json.loads(body)
      status_val = data.get('status', '')
      print(
        f'  Poll: status={status_val}, chunks={data.get("chunks_done")}/{data.get("chunks_total")}'
      )
      if status_val == 'ready':
        break
      if status_val == 'failed':
        break
      time.sleep(1)

    self._check('Job completed', status_val == 'ready', f'got {status_val}')

    if status_val == 'ready':
      audio_url = data.get('audio_url', f'/tts/{job_key}/audio')
      status, body, headers = _get(audio_url, base_url=self.base_url)
      ct = headers.get('Content-Type', headers.get('content-type', ''))
      self._check('Audio download returns 200', status == 200)
      self._check('Audio has content', len(body) > 100, f'{len(body)} bytes')
      out = OUTPUT_DIR / f'job.{"mp3" if "mpeg" in ct else "wav"}'
      out.write_bytes(body)
      print(f'  Saved: {out} ({len(body):,} bytes)')
    self._pause()

  def test_job_idempotent(self):
    print('\n--- 9. Job Idempotency ---')
    text = f'Idempotency test {int(time.time())}: same text yields same key.'
    _, body1, _ = _post_form(
      '/tts',
      {'text': text},
      headers={'Accept': 'application/json'},
      base_url=self.base_url,
    )
    _, body2, _ = _post_form(
      '/tts',
      {'text': text},
      headers={'Accept': 'application/json'},
      base_url=self.base_url,
    )
    key1 = json.loads(body1).get('key')
    key2 = json.loads(body2).get('key')
    self._check(
      'Same text produces same job key', key1 == key2, f'{key1[:16]} vs {key2[:16]}'
    )
    self._pause()

  def test_validation(self):
    print('\n--- 10. Validation ---')
    status, _body, _ = _post_form('/tts', {'text': ''}, base_url=self.base_url)
    self._check('Empty text returns 422', status == 422)

    status, _body, _ = _post_json(
      '/tts/stream', {'text': '   '}, base_url=self.base_url
    )
    self._check('Whitespace-only stream returns 422', status == 422)
    self._pause()

  def test_long_text(self, text_path: str):
    print(f'\n--- 11. Long Text ({text_path}) ---')
    text = Path(text_path).read_text()
    print(f'  Input: {len(text):,} chars')

    # Job-based
    print('\n  [Job-based]')
    start = time.perf_counter()
    status, body, _ = _post_form(
      '/tts',
      {'text': text},
      headers={'Accept': 'application/json'},
      base_url=self.base_url,
    )
    data = json.loads(body)
    self._check('Job submitted', status in (200, 202))
    job_key = data.get('key', '')
    print(f'  Job key: {job_key[:24]}...')

    if data.get('status') != 'ready':
      deadline = time.time() + 600  # 10 minute timeout for long text
      while time.time() < deadline:
        status, body, _ = _get(f'/tts/{job_key}', base_url=self.base_url)
        data = json.loads(body)
        st = data.get('status', '')
        chunks_done = data.get('chunks_done', 0)
        chunks_total = data.get('chunks_total', '?')
        print(f'  Poll: status={st}, progress={chunks_done}/{chunks_total}')
        if st in ('ready', 'failed'):
          break
        time.sleep(2)

    elapsed_job = time.perf_counter() - start
    self._check(
      'Long text job completed',
      data.get('status') == 'ready',
      f'got {data.get("status")}',
    )

    if data.get('status') == 'ready':
      audio_url = data.get('audio_url', f'/tts/{job_key}/audio')
      status, audio_body, headers = _get(audio_url, base_url=self.base_url)
      ct = headers.get('Content-Type', headers.get('content-type', ''))
      out = OUTPUT_DIR / f'long-job.{"mp3" if "mpeg" in ct else "wav"}'
      out.write_bytes(audio_body)
      print(f'  Saved: {out} ({len(audio_body):,} bytes, {elapsed_job:.1f}s total)')

    # Streaming
    print('\n  [Streaming]')
    start = time.perf_counter()
    try:
      status, body, headers = _post_json(
        '/tts/stream',
        {'text': text},
        base_url=self.base_url,
      )
      elapsed_stream = time.perf_counter() - start
      ct = headers.get('Content-Type', headers.get('content-type', ''))
      self._check('Long text stream returns 200', status == 200)
      self._check('Stream returns audio', len(body) > 1000, f'{len(body)} bytes')
      out = OUTPUT_DIR / 'long-stream.mp3'
      out.write_bytes(body)
      print(f'  Saved: {out} ({len(body):,} bytes, {elapsed_stream:.1f}s)')
    except Exception as exc:
      elapsed_stream = time.perf_counter() - start
      self._check('Long text stream completes', False, f'{type(exc).__name__}: {exc}')
      print(f'  Failed after {elapsed_stream:.1f}s')

    # Cache hit
    print('\n  [Cache hit]')
    start = time.perf_counter()
    status, body, _ = _post_form(
      '/tts',
      {'text': text},
      headers={'Accept': 'application/json'},
      base_url=self.base_url,
    )
    elapsed_cache = time.perf_counter() - start
    data = json.loads(body)
    self._check('Cache hit returns 200', status == 200)
    self._check('Status is ready (cached)', data.get('status') == 'ready')
    print(f'  Cache hit time: {elapsed_cache * 1000:.0f}ms')
    self._pause()

  def summary(self):
    total = self.passed + self.failed
    print(f'\n{"=" * 50}')
    print(f'Results: {self.passed}/{total} passed, {self.failed} failed')
    if self.failed:
      print('Some tests failed. Check output above.')
    else:
      print('All tests passed!')
    print(f'Audio files saved to: {OUTPUT_DIR}/')
    return self.failed == 0


def main():
  parser = argparse.ArgumentParser(description='Manual API test for tts-gateway')
  parser.add_argument(
    '--base-url', default=BASE_URL, help=f'Server URL (default: {BASE_URL})'
  )
  parser.add_argument(
    '--long-text', default=None, help='Path to long text file for stress test'
  )
  parser.add_argument('--step', action='store_true', help='Pause between tests')
  args = parser.parse_args()

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  # Check server is running
  try:
    urlopen(f'{args.base_url}/health', timeout=5)
  except (URLError, ConnectionRefusedError):
    print(f'ERROR: Server not reachable at {args.base_url}')
    print('Start it first: tts serve --provider kokoro --format mp3')
    sys.exit(1)

  runner = TestRunner(args.base_url, step=args.step)

  runner.test_health()
  runner.test_warmup()
  runner.test_legacy_tts()
  runner.test_legacy_tts_with_voice()
  runner.test_sync_endpoint()
  runner.test_streaming()
  job_key = runner.test_job_submit()
  runner.test_job_poll_and_download(job_key)
  runner.test_job_idempotent()
  runner.test_validation()

  if args.long_text:
    runner.test_long_text(args.long_text)

  ok = runner.summary()
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  main()
