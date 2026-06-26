"""Tests for scripts/verify_goal_artifacts.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  'verify_goal_artifacts',
  ROOT / 'scripts' / 'verify_goal_artifacts.py',
)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
sys.modules['verify_goal_artifacts'] = verify
SPEC.loader.exec_module(verify)


def _minimal_run(endpoint: str = '/tts/stream') -> dict[str, object]:
  return {
    'fixtureId': 'short',
    'endpoint': endpoint,
    'kind': 'stream',
    'iteration': 1,
    'status': 200,
    'ok': True,
    'contentType': 'audio/mpeg',
    'bytesRead': 100,
    'firstByteMs': 100.0,
    'totalMs': 200.0,
    'error': None,
  }


def _minimal_summary(endpoint: str = '/tts/stream') -> dict[str, object]:
  return {
    'fixtureId': 'short',
    'endpoint': endpoint,
    'kind': 'stream',
    'samples': 1,
    'okSamples': 1,
    'firstByteMsMedian': 100.0,
    'totalMsMedian': 200.0,
  }


def _minimal_benchmark(
  *,
  engine: str | None = None,
  compare_baseline: str | None = None,
  health: dict[str, object] | None = None,
  stream_endpoints: list[str] | None = None,
) -> dict[str, object]:
  endpoints = stream_endpoints or ['/tts/stream', '/tts/stream/pcm']
  payload: dict[str, object] = {
    'generatedAt': '2026-01-01T00:00:00Z',
    'baseUrl': 'http://127.0.0.1:45123',
    'condition': 'warm',
    'warmupRequested': True,
    'repeat': 1,
    'speechEndpoint': '/v1/speech',
    'streamEndpoints': endpoints,
    'fixtures': [
      {'id': 'short', 'label': 'Short', 'chars': 10, 'textHash': 'abc'},
    ],
    'runs': [_minimal_run(endpoint) for endpoint in endpoints],
    'summary': [_minimal_summary(endpoint) for endpoint in endpoints],
  }
  if engine is not None:
    payload['engine'] = engine
  if compare_baseline is not None:
    payload['compareBaseline'] = compare_baseline
  if health is not None:
    payload['health'] = health
  return payload


def _minimal_listening_report() -> str:
  return '\n'.join(
    [
      '# Stream Manual Listening Report',
      '',
      '## `/tts/stream`',
      '',
      '### Waveform sanity',
      '',
      '- status: PASS',
      '',
      '### Human listening verdict',
      '',
      '**Status:** PENDING — fill in after listening',
      '',
      '## `/tts/stream/pcm`',
      '',
      '### Waveform sanity',
      '',
      '- status: PASS',
      '',
      '### Human listening verdict',
      '',
      '**Status:** PENDING — fill in after listening',
      '',
      '## Overall human verdict',
      '',
      '**Status:** PENDING — review each endpoint',
      '',
    ]
  )


def _write_minimal_goal_dir(goal_dir: Path) -> None:
  goal_dir.mkdir(parents=True, exist_ok=True)
  for name in verify.REQUIRED_DOCS:
    (goal_dir / name).write_text(f'# {name}\n', encoding='utf-8')

  benchmark_specs = {
    'benchmarks/kokoro-baseline.json': _minimal_benchmark(
      stream_endpoints=['/tts/stream'],
    ),
    'benchmarks/stream-transport-comparison.json': _minimal_benchmark(),
    'benchmarks/kokoro-current-engine-metadata.json': _minimal_benchmark(
      engine='kokoro',
      health={'primaryEngine': 'kokoro'},
    ),
    'benchmarks/cosyvoice3-zero-shot-m1-short.json': _minimal_benchmark(
      engine='cosyvoice',
      compare_baseline='benchmarks/kokoro-current-engine-metadata.json',
    ),
    'benchmarks/cosyvoice3-zero-shot-m1-sentence.json': _minimal_benchmark(
      engine='cosyvoice',
      compare_baseline='benchmarks/kokoro-current-engine-metadata.json',
    ),
  }
  for rel_path, payload in benchmark_specs.items():
    path = goal_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')

  for rel_path in verify.LISTENING_REPORTS:
    path = goal_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_minimal_listening_report(), encoding='utf-8')

  asset_path = goal_dir / 'listening/assets/test/wav.wav'
  asset_path.parent.mkdir(parents=True, exist_ok=True)
  asset_path.write_bytes(b'RIFF')
  review_html = (
    '<!doctype html><html><body>'
    '<audio controls src="assets/test/wav.wav"></audio>'
    '</body></html>'
  )
  (goal_dir / 'listening/review.html').write_text(review_html, encoding='utf-8')


def test_complete_minimal_goal_dir_passes(tmp_path: Path) -> None:
  goal_dir = tmp_path / 'goal'
  _write_minimal_goal_dir(goal_dir)
  _, exit_code = verify.run_audit(goal_dir)
  assert exit_code == 0


def test_missing_human_verdict_warns_without_require_flag(tmp_path: Path) -> None:
  goal_dir = tmp_path / 'goal'
  _write_minimal_goal_dir(goal_dir)
  findings, exit_code = verify.run_audit(goal_dir)
  assert exit_code == 0
  verdict_findings = [
    finding for finding in findings if finding.check == 'listening:verdict.json'
  ]
  assert len(verdict_findings) == 1
  assert verdict_findings[0].level == 'WARN'


def test_missing_human_verdict_fails_with_require_flag(tmp_path: Path) -> None:
  goal_dir = tmp_path / 'goal'
  _write_minimal_goal_dir(goal_dir)
  _, exit_code = verify.run_audit(goal_dir, require_human_verdict=True)
  assert exit_code == 1


def test_missing_review_html_audio_fails(tmp_path: Path) -> None:
  goal_dir = tmp_path / 'goal'
  _write_minimal_goal_dir(goal_dir)
  (goal_dir / 'listening/assets/test/wav.wav').unlink()
  _, exit_code = verify.run_audit(goal_dir)
  assert exit_code == 1


def test_malformed_benchmark_json_fails(tmp_path: Path) -> None:
  goal_dir = tmp_path / 'goal'
  _write_minimal_goal_dir(goal_dir)
  bad_path = goal_dir / 'benchmarks/kokoro-baseline.json'
  bad_path.write_text('{not json', encoding='utf-8')
  _, exit_code = verify.run_audit(goal_dir)
  assert exit_code == 1


def test_pcm_listing_without_pcm_samples_fails(tmp_path: Path) -> None:
  goal_dir = tmp_path / 'goal'
  _write_minimal_goal_dir(goal_dir)
  benchmark_path = goal_dir / 'benchmarks/kokoro-current-engine-metadata.json'
  payload = json.loads(benchmark_path.read_text(encoding='utf-8'))
  payload['runs'] = [_minimal_run('/tts/stream')]
  payload['summary'] = [_minimal_summary('/tts/stream')]
  benchmark_path.write_text(json.dumps(payload), encoding='utf-8')

  findings, exit_code = verify.run_audit(goal_dir)

  assert exit_code == 1
  assert any(
    finding.check == 'benchmark:benchmarks/kokoro-current-engine-metadata.json:pcm'
    and 'runs missing endpoint /tts/stream/pcm' in finding.detail
    for finding in findings
  )


def test_main_returns_nonzero_for_missing_goal_dir(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  missing = tmp_path / 'missing-goal'
  exit_code = verify.main(['--goal-dir', str(missing)])
  captured = capsys.readouterr()
  assert exit_code == 1
  assert 'FAIL goal-dir' in captured.out
