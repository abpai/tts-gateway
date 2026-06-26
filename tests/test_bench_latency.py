"""Tests for scripts/bench_latency.py helpers and report assembly."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  'bench_latency',
  ROOT / 'scripts' / 'bench_latency.py',
)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules['bench_latency'] = bench
SPEC.loader.exec_module(bench)


class FakeResponse:
  """Minimal HTTP response for read_response tests."""

  def __init__(
    self,
    chunks: list[bytes],
    *,
    status: int = 200,
    content_type: str = 'audio/mpeg',
  ) -> None:
    self._chunks = list(chunks)
    self._index = 0
    self.status = status
    self._content_type = content_type

  def read(self, size: int) -> bytes:
    del size
    if self._index >= len(self._chunks):
      return b''
    chunk = self._chunks[self._index]
    self._index += 1
    return chunk

  def getheader(self, name: str) -> str | None:
    if name == 'content-type':
      return self._content_type
    return None


def test_fixture_texts_cover_required_categories() -> None:
  assert set(bench.FIXTURE_TEXTS) == {
    'short',
    'sentence',
    'medium',
    'long',
    'markdown',
  }


def test_fixture_from_text_hashes_and_counts() -> None:
  text = bench.FIXTURE_TEXTS['short']
  fixture = bench.fixture_from_text('short', text)
  assert fixture.id == 'short'
  assert fixture.chars == len(text)
  assert fixture.text_hash == bench.fixture_from_text('short', text).text_hash


def test_stream_endpoints_default() -> None:
  args = bench.parse_args([])
  assert bench.stream_endpoints(args) == ['/tts/stream', '/tts/stream/pcm']


def test_stream_endpoints_override_default() -> None:
  args = bench.parse_args(['--stream-endpoint', '/tts/stream/pcm'])
  assert bench.stream_endpoints(args) == ['/tts/stream/pcm']


def test_condition_helpers() -> None:
  warm = bench.parse_args(['--warmup'])
  assert bench.should_warmup(warm) is True
  assert bench.resolve_condition(warm) == 'warm'

  labeled = bench.parse_args(['--condition', 'cold'])
  assert bench.should_warmup(labeled) is False
  assert bench.resolve_condition(labeled) == 'cold'

  explicit_warm = bench.parse_args(['--condition', 'warm'])
  assert bench.should_warmup(explicit_warm) is True
  assert bench.resolve_condition(explicit_warm) == 'warm'


def test_cache_bust_helpers() -> None:
  plain = bench.parse_args([])
  assert bench.resolve_cache_bust_token(plain) is None
  assert bench.apply_cache_bust('hello', None) == 'hello'

  manual = bench.parse_args(['--cache-bust-token', 'run-1'])
  assert bench.resolve_cache_bust_token(manual) == 'run-1'
  assert bench.apply_cache_bust('hello', 'run-1').endswith('run-1.')

  generated = bench.resolve_cache_bust_token(bench.parse_args(['--cache-bust']))
  assert generated is not None
  assert generated.endswith('Z')


def test_make_body_formats() -> None:
  speech_body, speech_headers = bench.make_body('speech', 'hello')
  assert speech_body == b'text=hello'
  assert speech_headers['Content-Type'] == 'application/x-www-form-urlencoded'

  stream_body, stream_headers = bench.make_body('stream', 'hello')
  assert json.loads(stream_body.decode()) == {'text': 'hello'}
  assert stream_headers['Content-Type'] == 'application/json'


def test_elapsed_ms_rounds() -> None:
  assert bench.elapsed_ms(1.0, None) is None
  assert bench.elapsed_ms(1.0, 1.0015) == 1.5


def test_read_response_tracks_first_byte_and_total() -> None:
  response = FakeResponse([b'abc', b'def'])
  with patch.object(bench.time, 'perf_counter', side_effect=[0.01, 0.03, 0.05]):
    timing = bench.read_response(response, started=0.0)

  assert timing['status'] == 200
  assert timing['bytes_read'] == 6
  assert timing['first_byte_ms'] == 10.0
  assert timing['total_ms'] == 50.0
  assert timing['content_type'] == 'audio/mpeg'


def test_median_or_none() -> None:
  assert bench.median_or_none([]) is None
  assert bench.median_or_none([10.0, 20.0, 30.0]) == 20.0


def test_summarize_groups_by_fixture_endpoint_and_kind() -> None:
  runs = [
    bench.EndpointTiming(
      fixtureId='short',
      endpoint='/v1/speech',
      kind='speech',
      iteration=1,
      status=200,
      ok=True,
      contentType='audio/mpeg',
      bytesRead=100,
      firstByteMs=5.0,
      totalMs=50.0,
    ),
    bench.EndpointTiming(
      fixtureId='short',
      endpoint='/tts/stream',
      kind='stream',
      iteration=1,
      status=200,
      ok=True,
      contentType='audio/mpeg',
      bytesRead=100,
      firstByteMs=10.0,
      totalMs=80.0,
    ),
  ]
  summary = bench.summarize(runs)
  assert len(summary) == 2
  speech = next(item for item in summary if item.kind == 'speech')
  stream = next(item for item in summary if item.kind == 'stream')
  assert speech.total_ms_median == 50.0
  assert stream.first_byte_ms_median == 10.0


def test_compare_reports_emits_deltas() -> None:
  baseline = bench.BenchmarkReport(
    generatedAt='2026-01-01T00:00:00+00:00',
    baseUrl='http://127.0.0.1:45123',
    condition='warm',
    warmupRequested=True,
    repeat=1,
    speechEndpoint='/v1/speech',
    streamEndpoints=['/tts/stream'],
    fixtures=[],
    runs=[],
    summary=[
      bench.EndpointSummary(
        fixtureId='short',
        endpoint='/tts/stream',
        kind='stream',
        samples=1,
        okSamples=1,
        firstByteMsMedian=100.0,
        totalMsMedian=200.0,
      )
    ],
  )
  current = baseline.model_copy(
    update={
      'summary': [
        bench.EndpointSummary(
          fixtureId='short',
          endpoint='/tts/stream',
          kind='stream',
          samples=1,
          okSamples=1,
          firstByteMsMedian=120.0,
          totalMsMedian=250.0,
        )
      ]
    }
  )
  comparisons = bench.compare_reports(current, baseline)
  assert len(comparisons) == 1
  assert comparisons[0].first_byte_delta_ms == 20.0
  assert comparisons[0].total_delta_ms == 50.0


def test_parse_args_engine_options() -> None:
  args = bench.parse_args(['--engine', 'cosyvoice', '--require-engine-match'])
  assert args.engine == 'cosyvoice'
  assert args.require_engine_match is True


def test_fetch_health_parses_payload() -> None:
  health_body = json.dumps(
    {
      'ok': True,
      'primaryEngine': 'kokoro',
      'fallbackEngine': 'pocket',
      'engineChain': ['kokoro', 'pocket'],
      'streamFirstChunkMaxChars': 80,
      'streamChunkMaxChars': 240,
      'engines': {'kokoro': {'mode': 'native'}},
    }
  ).encode()

  class FakeHealthResponse:
    status = 200

    def read(self) -> bytes:
      return health_body

  class FakeConn:
    def request(self, method: str, path: str) -> None:
      assert method == 'GET'
      assert path == '/health'

    def getresponse(self) -> FakeHealthResponse:
      return FakeHealthResponse()

    def close(self) -> None:
      return None

  with patch.object(bench, 'open_connection', return_value=FakeConn()):
    snapshot = bench.fetch_health('http://127.0.0.1:45123')

  assert snapshot['ok'] is True
  assert snapshot['status'] == 200
  assert snapshot['primaryEngine'] == 'kokoro'
  assert snapshot['fallbackEngine'] == 'pocket'
  assert snapshot['engineChain'] == ['kokoro', 'pocket']
  assert snapshot['streamFirstChunkMaxChars'] == 80
  assert snapshot['streamChunkMaxChars'] == 240
  assert snapshot['engines'] == {'kokoro': {'mode': 'native'}}


def test_fetch_health_records_network_error() -> None:
  with patch.object(bench, 'open_connection', side_effect=OSError('refused')):
    snapshot = bench.fetch_health('http://127.0.0.1:45123')

  assert snapshot['ok'] is False
  assert snapshot['error'] == 'OSError: refused'


def test_engine_mismatch_warning() -> None:
  health = {'ok': True, 'primaryEngine': 'kokoro'}
  warning = bench.engine_mismatch_warning('cosyvoice', health)
  assert warning is not None
  assert 'cosyvoice' in warning
  assert 'kokoro' in warning
  assert bench.engine_mismatch_warning('kokoro', health) is None
  assert bench.engine_mismatch_warning('kokoro', {'ok': False}) is None


def test_load_report_accepts_legacy_json_without_engine_metadata() -> None:
  legacy = {
    'generatedAt': '2026-01-01T00:00:00+00:00',
    'baseUrl': 'http://127.0.0.1:45123',
    'condition': 'warm',
    'warmupRequested': True,
    'repeat': 1,
    'speechEndpoint': '/v1/speech',
    'streamEndpoints': ['/tts/stream'],
    'fixtures': [],
    'runs': [],
    'summary': [],
  }
  report = bench.BenchmarkReport.model_validate(legacy)
  assert report.engine is None
  assert report.health is None
  assert report.warnings == []


def test_build_report_with_mocked_http(tmp_path: Path) -> None:
  args = argparse.Namespace(
    base_url='http://127.0.0.1:45123',
    fixture=['short'],
    repeat=1,
    speech_endpoint='/v1/speech',
    stream_endpoint=['/tts/stream', '/tts/stream/pcm'],
    warmup=False,
    condition='as-is',
    cache_bust=False,
    cache_bust_token=None,
  )

  def fake_post(base_url: str, path: str, body: bytes, headers: dict[str, str]) -> dict:
    del base_url, body, headers
    if path == '/warmup':
      return {
        'status': 200,
        'content_type': 'application/json',
        'bytes_read': 2,
        'first_byte_ms': 1.0,
        'total_ms': 2.0,
      }
    return {
      'status': 200,
      'content_type': 'audio/mpeg',
      'bytes_read': 128,
      'first_byte_ms': 10.0 if 'stream' in path else 50.0,
      'total_ms': 100.0 if path == '/v1/speech' else 200.0,
    }

  with patch.object(bench, 'post_timed', side_effect=fake_post):
    report = bench.build_report(
      args,
      health={'ok': True, 'primaryEngine': 'kokoro'},
      warnings=[],
    )

  assert report.condition == 'as-is'
  assert report.engine is None
  assert report.health == {'ok': True, 'primaryEngine': 'kokoro'}
  assert report.cache_bust_token is None
  assert report.stream_endpoints == ['/tts/stream', '/tts/stream/pcm']
  assert len(report.fixtures) == 1
  assert len(report.runs) == 3
  assert len(report.summary) == 3


def test_build_report_records_cache_bust_token() -> None:
  args = argparse.Namespace(
    base_url='http://127.0.0.1:45123',
    fixture=['short'],
    repeat=1,
    speech_endpoint='/v1/speech',
    stream_endpoint=None,
    warmup=False,
    condition='as-is',
    cache_bust=False,
    cache_bust_token='baseline-1',
  )

  with patch.object(bench, 'post_timed') as post_timed:
    post_timed.return_value = {
      'status': 200,
      'content_type': 'audio/mpeg',
      'bytes_read': 128,
      'first_byte_ms': 10.0,
      'total_ms': 20.0,
    }
    report = bench.build_report(
      args,
      health={'ok': True, 'primaryEngine': 'kokoro'},
      warnings=[],
    )

  assert report.cache_bust_token == 'baseline-1'
  assert report.fixtures[0].chars > len(bench.FIXTURE_TEXTS['short'])
  assert len(report.runs) == 3


def test_main_writes_json_and_compare_metadata(tmp_path: Path) -> None:
  baseline_path = tmp_path / 'baseline.json'
  output_path = tmp_path / 'current.json'
  baseline = bench.BenchmarkReport(
    generatedAt='2026-01-01T00:00:00+00:00',
    baseUrl='http://127.0.0.1:45123',
    condition='warm',
    warmupRequested=True,
    repeat=1,
    speechEndpoint='/v1/speech',
    streamEndpoints=['/tts/stream'],
    fixtures=[bench.fixture_from_text('short', bench.FIXTURE_TEXTS['short'])],
    runs=[],
    summary=[
      bench.EndpointSummary(
        fixtureId='short',
        endpoint='/v1/speech',
        kind='speech',
        samples=1,
        okSamples=1,
        firstByteMsMedian=None,
        totalMsMedian=100.0,
      )
    ],
  )
  baseline_path.write_text(baseline.model_dump_json(by_alias=True))

  def fake_post(base_url: str, path: str, body: bytes, headers: dict[str, str]) -> dict:
    del base_url, body, headers
    return {
      'status': 200,
      'content_type': 'audio/mpeg',
      'bytes_read': 64,
      'first_byte_ms': 5.0,
      'total_ms': 110.0,
    }

  stdout = StringIO()
  with patch.object(
    bench,
    'fetch_health',
    return_value={'ok': True, 'primaryEngine': 'kokoro'},
  ):
    with patch.object(bench, 'post_timed', side_effect=fake_post):
      with patch('sys.stdout', stdout):
        exit_code = bench.main(
          [
            '--fixture',
            'short',
            '--output',
            str(output_path),
            '--compare',
            str(baseline_path),
          ]
        )

  assert exit_code == 0
  payload = json.loads(output_path.read_text())
  assert payload['compareBaseline'] == str(baseline_path)
  assert payload['comparisons'][0]['totalDeltaMs'] == 10.0
  assert payload['health']['primaryEngine'] == 'kokoro'
  assert payload['engine'] is None
  assert payload['warnings'] == []


def test_main_records_engine_mismatch_warning(tmp_path: Path) -> None:
  output_path = tmp_path / 'current.json'

  def fake_post(base_url: str, path: str, body: bytes, headers: dict[str, str]) -> dict:
    del base_url, body, headers
    return {
      'status': 200,
      'content_type': 'audio/mpeg',
      'bytes_read': 64,
      'first_byte_ms': 5.0,
      'total_ms': 110.0,
    }

  stdout = StringIO()
  with patch.object(
    bench,
    'fetch_health',
    return_value={'ok': True, 'primaryEngine': 'kokoro'},
  ):
    with patch.object(bench, 'post_timed', side_effect=fake_post):
      with patch('sys.stdout', stdout):
        exit_code = bench.main(
          [
            '--fixture',
            'short',
            '--engine',
            'cosyvoice',
            '--output',
            str(output_path),
          ]
        )

  assert exit_code == 0
  payload = json.loads(output_path.read_text())
  assert payload['engine'] == 'cosyvoice'
  assert len(payload['warnings']) == 1
  assert 'cosyvoice' in payload['warnings'][0]
  assert 'kokoro' in payload['warnings'][0]
  assert 'warnings' in stdout.getvalue()


def test_main_require_engine_match_exits_before_benchmarks() -> None:
  with patch.object(
    bench,
    'fetch_health',
    return_value={'ok': True, 'primaryEngine': 'kokoro'},
  ):
    with patch.object(bench, 'post_timed') as post_timed:
      exit_code = bench.main(['--engine', 'cosyvoice', '--require-engine-match'])

  assert exit_code == 1
  post_timed.assert_not_called()


def test_main_rejects_non_positive_repeat() -> None:
  with pytest.raises(SystemExit):
    bench.main(['--repeat', '0'])
