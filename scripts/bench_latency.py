#!/usr/bin/env python3
"""Benchmark TTS endpoint latency and emit JSON results."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import statistics
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel, Field

EndpointKind = Literal['speech', 'stream']
RunCondition = Literal['as-is', 'warm', 'cold']
EngineKind = Literal['kokoro', 'pocket', 'cosyvoice']
NETWORK_ERRORS = (http.client.HTTPException, OSError, TimeoutError)


class Fixture(BaseModel):
  """Text fixture used for latency measurement."""

  id: str
  label: str
  chars: int
  text_hash: str = Field(alias='textHash')


class EndpointTiming(BaseModel):
  """Single endpoint timing sample."""

  fixture_id: str = Field(alias='fixtureId')
  endpoint: str
  kind: EndpointKind
  iteration: int
  status: int
  ok: bool
  content_type: str | None = Field(alias='contentType')
  bytes_read: int = Field(alias='bytesRead')
  first_byte_ms: float | None = Field(alias='firstByteMs')
  total_ms: float = Field(alias='totalMs')
  error: str | None = None


class EndpointSummary(BaseModel):
  """Summary statistics for one fixture and endpoint."""

  fixture_id: str = Field(alias='fixtureId')
  endpoint: str
  kind: EndpointKind
  samples: int
  ok_samples: int = Field(alias='okSamples')
  first_byte_ms_median: float | None = Field(alias='firstByteMsMedian')
  total_ms_median: float | None = Field(alias='totalMsMedian')


class Comparison(BaseModel):
  """Delta between a current summary and a baseline summary."""

  fixture_id: str = Field(alias='fixtureId')
  endpoint: str
  kind: EndpointKind
  first_byte_delta_ms: float | None = Field(alias='firstByteDeltaMs')
  total_delta_ms: float | None = Field(alias='totalDeltaMs')


class BenchmarkReport(BaseModel):
  """Benchmark run report."""

  generated_at: str = Field(alias='generatedAt')
  base_url: str = Field(alias='baseUrl')
  engine: str | None = None
  condition: RunCondition
  warmup_requested: bool = Field(alias='warmupRequested')
  cache_bust_token: str | None = Field(default=None, alias='cacheBustToken')
  repeat: int
  speech_endpoint: str = Field(alias='speechEndpoint')
  stream_endpoints: list[str] = Field(alias='streamEndpoints')
  compare_baseline: str | None = Field(default=None, alias='compareBaseline')
  health: dict[str, Any] | None = None
  warnings: list[str] = Field(default_factory=list)
  fixtures: list[Fixture]
  runs: list[EndpointTiming]
  summary: list[EndpointSummary]
  comparisons: list[Comparison] = Field(default_factory=list)
  warmup: dict[str, Any] | None = None


FIXTURE_TEXTS = {
  'short': 'Short latency check.',
  'sentence': (
    'This is a slightly longer latency check for the Raycast text to speech '
    'reader. It should be similar to a selected sentence or two.'
  ),
  'medium': (
    'First sentence for streaming. Second sentence arrives next. Third sentence '
    'wraps it up. Fourth sentence keeps enough material in the sample to exercise '
    'the stream path without becoming an article-sized request.'
  ),
  'long': (
    'This is a longer paragraph selected from an article. It has enough words to '
    'exercise chunking and make the difference between buffered file generation '
    'and progressive playback visible. ' * 12
  ).strip(),
  'markdown': (
    '# Notes\n\n'
    '- Read [the guide](https://example.com/guide) before shipping.\n'
    '- Keep `Option+R` fast for selected text.\n'
    'Raw URL: https://example.com/noisy/path?with=query\n'
    '**Bold thought:** streaming should start before the whole file exists.'
  ),
}


def fixture_from_text(fixture_id: str, text: str) -> Fixture:
  digest = hashlib.sha256(text.encode()).hexdigest()
  return Fixture(
    id=fixture_id,
    label=fixture_id.replace('_', ' ').title(),
    chars=len(text),
    textHash=digest,
  )


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--base-url', default='http://127.0.0.1:45123')
  parser.add_argument('--output', type=Path)
  parser.add_argument('--compare', type=Path)
  parser.add_argument('--repeat', type=int, default=1)
  parser.add_argument('--condition', choices=['as-is', 'warm', 'cold'], default='as-is')
  parser.add_argument('--warmup', action='store_true')
  parser.add_argument('--cache-bust', action='store_true')
  parser.add_argument('--cache-bust-token')
  parser.add_argument('--speech-endpoint', default='/v1/speech')
  parser.add_argument('--stream-endpoint', action='append')
  parser.add_argument('--fixture', action='append', choices=sorted(FIXTURE_TEXTS))
  parser.add_argument('--engine', choices=['kokoro', 'pocket', 'cosyvoice'])
  parser.add_argument(
    '--require-engine-match',
    action='store_true',
    help='Exit before benchmarks when --engine disagrees with /health primaryEngine',
  )
  return parser.parse_args(argv)


def stream_endpoints(args: argparse.Namespace) -> list[str]:
  return args.stream_endpoint or ['/tts/stream', '/tts/stream/pcm']


def should_warmup(args: argparse.Namespace) -> bool:
  return bool(args.warmup or args.condition == 'warm')


def resolve_condition(args: argparse.Namespace) -> RunCondition:
  if args.warmup:
    return 'warm'
  return args.condition


def resolve_cache_bust_token(args: argparse.Namespace) -> str | None:
  if args.cache_bust_token:
    return str(args.cache_bust_token)
  if args.cache_bust:
    return datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
  return None


def apply_cache_bust(text: str, token: str | None) -> str:
  if token is None:
    return text
  return f'{text}\n\nBenchmark cache bust token: {token}.'


def request_path_for(base_url: str, path: str) -> str:
  parsed = urlparse(base_url)
  request_path = f'{parsed.path.rstrip("/")}{path}'
  return request_path or '/'


def open_connection(base_url: str) -> http.client.HTTPConnection:
  parsed = urlparse(base_url)
  conn_cls = (
    http.client.HTTPSConnection
    if parsed.scheme == 'https'
    else http.client.HTTPConnection
  )
  return conn_cls(parsed.hostname or '', parsed.port, timeout=180)


def fetch_health(base_url: str) -> dict[str, Any]:
  request_path = request_path_for(base_url, '/health')
  try:
    conn = open_connection(base_url)
  except NETWORK_ERRORS as exc:
    return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
  try:
    conn.request('GET', request_path)
    response = conn.getresponse()
    status = response.status
    body = response.read()
    if not 200 <= status < 300:
      return {'ok': False, 'status': status}
    payload = json.loads(body.decode())
    snapshot: dict[str, Any] = {
      'ok': True,
      'status': status,
      'primaryEngine': payload.get('primaryEngine'),
      'fallbackEngine': payload.get('fallbackEngine'),
      'engineChain': payload.get('engineChain'),
      'streamFirstChunkMaxChars': payload.get('streamFirstChunkMaxChars'),
      'streamChunkMaxChars': payload.get('streamChunkMaxChars'),
    }
    if 'engines' in payload:
      snapshot['engines'] = payload['engines']
    return snapshot
  except json.JSONDecodeError as exc:
    return {'ok': False, 'status': status, 'error': f'JSONDecodeError: {exc}'}
  except NETWORK_ERRORS as exc:
    return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
  finally:
    conn.close()


def engine_mismatch_warning(requested: str, health: dict[str, Any]) -> str | None:
  if not health.get('ok'):
    return None
  primary = health.get('primaryEngine')
  if primary is None or primary == requested:
    return None
  return f'Requested engine {requested!r} but gateway primaryEngine is {primary!r}'


def post_timed(base_url: str, path: str, body: bytes, headers: dict[str, str]) -> dict:
  request_path = request_path_for(base_url, path)
  conn = open_connection(base_url)
  started = time.perf_counter()
  try:
    conn.request('POST', request_path, body=body, headers=headers)
    response = conn.getresponse()
    return read_response(response, started)
  finally:
    conn.close()


def read_response(response: http.client.HTTPResponse, started: float) -> dict:
  first_byte_at: float | None = None
  bytes_read = 0
  while True:
    chunk = response.read(8192)
    now = time.perf_counter()
    if chunk and first_byte_at is None:
      first_byte_at = now
    if not chunk:
      return {
        'status': response.status,
        'content_type': response.getheader('content-type'),
        'bytes_read': bytes_read,
        'first_byte_ms': elapsed_ms(started, first_byte_at),
        'total_ms': elapsed_ms(started, now),
      }
    bytes_read += len(chunk)


def elapsed_ms(started: float, ended: float | None) -> float | None:
  if ended is None:
    return None
  return round((ended - started) * 1000, 3)


def make_body(kind: EndpointKind, text: str) -> tuple[bytes, dict[str, str]]:
  if kind == 'speech':
    return urlencode({'text': text}).encode(), {
      'Content-Type': 'application/x-www-form-urlencoded'
    }
  return json.dumps({'text': text}).encode(), {'Content-Type': 'application/json'}


def run_one(
  base_url: str,
  path: str,
  kind: EndpointKind,
  fixture_id: str,
  text: str,
  iteration: int,
) -> EndpointTiming:
  body, headers = make_body(kind, text)
  try:
    timing = post_timed(base_url, path, body, headers)
    status = int(timing['status'])
    return EndpointTiming(
      fixtureId=fixture_id,
      endpoint=path,
      kind=kind,
      iteration=iteration,
      status=status,
      ok=200 <= status < 300,
      contentType=timing['content_type'],
      bytesRead=timing['bytes_read'],
      firstByteMs=timing['first_byte_ms'],
      totalMs=timing['total_ms'],
    )
  except NETWORK_ERRORS as exc:
    return failed_timing(path, kind, fixture_id, iteration, exc)


def failed_timing(
  path: str, kind: EndpointKind, fixture_id: str, iteration: int, exc: Exception
) -> EndpointTiming:
  return EndpointTiming(
    fixtureId=fixture_id,
    endpoint=path,
    kind=kind,
    iteration=iteration,
    status=0,
    ok=False,
    contentType=None,
    bytesRead=0,
    firstByteMs=None,
    totalMs=0,
    error=f'{type(exc).__name__}: {exc}',
  )


def summarize(runs: Iterable[EndpointTiming]) -> list[EndpointSummary]:
  groups: dict[tuple[str, str, EndpointKind], list[EndpointTiming]] = {}
  for run in runs:
    groups.setdefault((run.fixture_id, run.endpoint, run.kind), []).append(run)
  return [summary_for(key, values) for key, values in sorted(groups.items())]


def summary_for(
  key: tuple[str, str, EndpointKind], runs: list[EndpointTiming]
) -> EndpointSummary:
  fixture_id, endpoint, kind = key
  ok_runs = [run for run in runs if run.ok]
  first_values = [run.first_byte_ms for run in ok_runs if run.first_byte_ms is not None]
  total_values = [run.total_ms for run in ok_runs]
  return EndpointSummary(
    fixtureId=fixture_id,
    endpoint=endpoint,
    kind=kind,
    samples=len(runs),
    okSamples=len(ok_runs),
    firstByteMsMedian=median_or_none(first_values),
    totalMsMedian=median_or_none(total_values),
  )


def median_or_none(values: list[float]) -> float | None:
  if not values:
    return None
  return round(float(statistics.median(values)), 3)


def run_warmup(base_url: str) -> dict[str, Any]:
  try:
    timing = post_timed(base_url, '/warmup', b'', {})
    return {'ok': 200 <= timing['status'] < 300, **timing}
  except NETWORK_ERRORS as exc:
    return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


def compare_reports(
  current: BenchmarkReport, baseline: BenchmarkReport
) -> list[Comparison]:
  baseline_map = {
    (item.fixture_id, item.endpoint, item.kind): item for item in baseline.summary
  }
  return [
    compare_summary(item, baseline_map[(item.fixture_id, item.endpoint, item.kind)])
    for item in current.summary
    if (item.fixture_id, item.endpoint, item.kind) in baseline_map
  ]


def compare_summary(current: EndpointSummary, baseline: EndpointSummary) -> Comparison:
  return Comparison(
    fixtureId=current.fixture_id,
    endpoint=current.endpoint,
    kind=current.kind,
    firstByteDeltaMs=delta(current.first_byte_ms_median, baseline.first_byte_ms_median),
    totalDeltaMs=delta(current.total_ms_median, baseline.total_ms_median),
  )


def delta(current: float | None, baseline: float | None) -> float | None:
  if current is None or baseline is None:
    return None
  return round(current - baseline, 3)


def build_report(
  args: argparse.Namespace,
  *,
  health: dict[str, Any],
  warnings: list[str],
) -> BenchmarkReport:
  selected_ids = args.fixture or list(FIXTURE_TEXTS)
  cache_bust_token = resolve_cache_bust_token(args)
  fixture_texts = {
    fixture_id: apply_cache_bust(FIXTURE_TEXTS[fixture_id], cache_bust_token)
    for fixture_id in selected_ids
  }
  endpoints = stream_endpoints(args)
  warmup = run_warmup(args.base_url) if should_warmup(args) else None
  runs = collect_runs(args, fixture_texts, endpoints)
  return BenchmarkReport(
    generatedAt=datetime.now(UTC).isoformat(),
    baseUrl=args.base_url,
    engine=getattr(args, 'engine', None),
    condition=resolve_condition(args),
    warmupRequested=should_warmup(args),
    cacheBustToken=cache_bust_token,
    repeat=args.repeat,
    speechEndpoint=args.speech_endpoint,
    streamEndpoints=endpoints,
    health=health,
    warnings=warnings,
    fixtures=[
      fixture_from_text(fixture_id, text) for fixture_id, text in fixture_texts.items()
    ],
    runs=runs,
    summary=summarize(runs),
    warmup=warmup,
  )


def collect_runs(
  args: argparse.Namespace,
  fixture_texts: dict[str, str],
  endpoints: list[str],
) -> list[EndpointTiming]:
  runs: list[EndpointTiming] = []
  for iteration in range(1, args.repeat + 1):
    for fixture_id, text in fixture_texts.items():
      runs.append(
        run_one(
          args.base_url, args.speech_endpoint, 'speech', fixture_id, text, iteration
        )
      )
      for endpoint in endpoints:
        runs.append(
          run_one(args.base_url, endpoint, 'stream', fixture_id, text, iteration)
        )
  return runs


def load_report(path: Path) -> BenchmarkReport:
  return BenchmarkReport.model_validate_json(path.read_text())


def write_report(report: BenchmarkReport, output: Path | None) -> None:
  payload = report.model_dump_json(by_alias=True, indent=2)
  if output is None:
    print(payload)
    return
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(f'{payload}\n')


def print_summary(report: BenchmarkReport) -> None:
  if report.warnings:
    print('warnings')
    for warning in report.warnings:
      print(warning)
    print()
  print('fixture endpoint kind first_byte_ms total_ms ok/samples')
  for item in report.summary:
    print(
      item.fixture_id,
      item.endpoint,
      item.kind,
      item.first_byte_ms_median,
      item.total_ms_median,
      f'{item.ok_samples}/{item.samples}',
    )
  if report.comparisons:
    print('\ncomparison deltas vs baseline')
    for item in report.comparisons:
      print(
        item.fixture_id, item.endpoint, item.first_byte_delta_ms, item.total_delta_ms
      )


def main(argv: list[str] | None = None) -> int:
  args = parse_args(sys.argv[1:] if argv is None else argv)
  if args.repeat <= 0:
    raise SystemExit('--repeat must be > 0')
  health = fetch_health(args.base_url)
  warnings: list[str] = []
  if args.engine is not None:
    mismatch = engine_mismatch_warning(args.engine, health)
    if mismatch is not None:
      if args.require_engine_match:
        print(mismatch, file=sys.stderr)
        return 1
      warnings.append(mismatch)
  report = build_report(args, health=health, warnings=warnings)
  if args.compare:
    report.compare_baseline = str(args.compare)
    report.comparisons = compare_reports(report, load_report(args.compare))
  write_report(report, args.output)
  if args.output is not None:
    print_summary(report)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
