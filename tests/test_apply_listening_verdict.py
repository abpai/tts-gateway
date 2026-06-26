"""Tests for scripts/apply_listening_verdict.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  'apply_listening_verdict',
  ROOT / 'scripts' / 'apply_listening_verdict.py',
)
assert SPEC is not None and SPEC.loader is not None
verdict_mod = importlib.util.module_from_spec(SPEC)
sys.modules['apply_listening_verdict'] = verdict_mod
SPEC.loader.exec_module(verdict_mod)


def _sample_payload(**overrides: bool | str) -> dict[str, bool | str]:
  payload: dict[str, bool | str] = {
    'reviewedAt': '2026-06-26T19:00:00.000Z',
    'notes': 'Looks good in Raycast.',
    'clicks': True,
    'gaps': True,
    'prompt': True,
    'stop': True,
    'option-r': True,
    'kokoro-default': True,
  }
  payload.update(overrides)
  return payload


def _sample_report() -> str:
  return '\n'.join(
    [
      '# Stream Manual Listening Report',
      '',
      '## `/tts/stream`',
      '',
      '### Human listening verdict',
      '',
      '**Status:** PENDING — fill in after listening',
      '',
      '- [ ] No clicks or pops at chunk boundaries',
      '',
      '## `/tts/stream/pcm`',
      '',
      '### Human listening verdict',
      '',
      '**Status:** PENDING — fill in after listening',
      '',
      '- [ ] No clicks or pops at chunk boundaries',
      '',
      '## Overall human verdict',
      '',
      (
        '**Status:** PENDING — review each endpoint with the replay commands '
        'above, then check items in Raycast (stop/cancel and Option+R).'
      ),
      '',
    ]
  )


def test_read_input_json_rejects_invalid(tmp_path: Path) -> None:
  bad = tmp_path / 'bad.json'
  bad.write_text('not json', encoding='utf-8')
  with pytest.raises(verdict_mod.VerdictValidationError, match='Invalid JSON'):
    verdict_mod.read_input_json(bad)


def test_validate_rejects_non_object(tmp_path: Path) -> None:
  payload_path = tmp_path / 'array.json'
  payload_path.write_text('[1, 2]', encoding='utf-8')
  with pytest.raises(
    verdict_mod.VerdictValidationError,
    match='Verdict JSON must be an object',
  ):
    verdict_mod.read_input_json(payload_path)


def test_validate_rejects_missing_boolean() -> None:
  payload = _sample_payload()
  del payload['clicks']
  with pytest.raises(
    verdict_mod.VerdictValidationError,
    match='Missing required boolean checks: clicks',
  ):
    verdict_mod.validate_verdict_input(payload)


def test_validate_rejects_non_boolean_check() -> None:
  payload = _sample_payload(clicks='yes')
  with pytest.raises(
    verdict_mod.VerdictValidationError,
    match='Required checks must be booleans: clicks',
  ):
    verdict_mod.validate_verdict_input(payload)


def test_validate_rejects_blank_reviewed_at() -> None:
  payload = _sample_payload(reviewedAt='   ')
  with pytest.raises(
    verdict_mod.VerdictValidationError,
    match='reviewedAt is required',
  ):
    verdict_mod.validate_verdict_input(payload)


def test_build_output_verdict_pass() -> None:
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)
  output = verdict_mod.build_output_verdict(
    _sample_payload(),
    applied_at=applied_at,
  )
  assert output['complete'] is True
  assert output['status'] == 'PASS'
  assert output['missingChecks'] == []
  assert output['appliedAt'] == '2026-06-26T19:05:00Z'


def test_build_output_verdict_partial() -> None:
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)
  output = verdict_mod.build_output_verdict(
    _sample_payload(clicks=False, stop=False),
    applied_at=applied_at,
  )
  assert output['complete'] is False
  assert output['status'] == 'PARTIAL'
  assert output['missingChecks'] == ['clicks', 'stop']


def test_update_report_pass_updates_overall_and_endpoints() -> None:
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)
  verdict = verdict_mod.build_output_verdict(
    _sample_payload(),
    applied_at=applied_at,
  )
  updated = verdict_mod.update_report_content(_sample_report(), verdict)
  assert '**Status:** PASS — human listening checklist complete' in updated
  assert updated.count('**Status:** PASS — human listening checklist complete') == 3
  assert '**Status:** **Status:**' not in updated
  assert 'PENDING' not in updated
  assert '## Recorded human verdict' in updated
  assert '`2026-06-26T19:00:00.000Z`' in updated
  assert 'Looks good in Raycast.' in updated


def test_update_report_partial_updates_overall_only() -> None:
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)
  verdict = verdict_mod.build_output_verdict(
    _sample_payload(gaps=False),
    applied_at=applied_at,
  )
  updated = verdict_mod.update_report_content(_sample_report(), verdict)
  assert '**Status:** PARTIAL — missing checks:' in updated
  assert '**Status:** **Status:**' not in updated
  assert 'No audible gaps or dropped audio between chunks' in updated
  assert updated.count('**Status:** PENDING — fill in after listening') == 2
  assert updated.count('## Recorded human verdict') == 1


def test_update_report_requires_overall_status() -> None:
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)
  verdict = verdict_mod.build_output_verdict(
    _sample_payload(),
    applied_at=applied_at,
  )
  report = _sample_report().replace('## Overall human verdict', '## Summary')

  with pytest.raises(
    verdict_mod.ReportUpdateError,
    match='expected exactly one overall verdict status',
  ):
    verdict_mod.update_report_content(report, verdict)


def test_update_report_pass_requires_endpoint_statuses() -> None:
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)
  verdict = verdict_mod.build_output_verdict(
    _sample_payload(),
    applied_at=applied_at,
  )
  report = _sample_report().replace(
    '### Human listening verdict\n\n**Status:** PENDING',
    '### Human listening verdict\n\nNo status here',
    1,
  )

  with pytest.raises(
    verdict_mod.ReportUpdateError,
    match='expected every endpoint listening verdict to include a status',
  ):
    verdict_mod.update_report_content(report, verdict)


def test_update_report_is_idempotent_for_recorded_section() -> None:
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)
  verdict = verdict_mod.build_output_verdict(
    _sample_payload(notes=''),
    applied_at=applied_at,
  )
  first = verdict_mod.update_report_content(_sample_report(), verdict)
  second = verdict_mod.update_report_content(first, verdict)
  assert first == second
  assert second.count('## Recorded human verdict') == 1
  assert '- notes:' not in second


def test_apply_verdict_writes_files(tmp_path: Path) -> None:
  report_path = tmp_path / 'report.md'
  report_path.write_text(_sample_report(), encoding='utf-8')
  output_path = tmp_path / 'verdict.json'
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)

  result = verdict_mod.apply_verdict(
    _sample_payload(stop=False),
    output_path=output_path,
    report_paths=(report_path,),
    applied_at=applied_at,
    dry_run=False,
  )

  assert result['verdict']['status'] == 'PARTIAL'
  saved = json.loads(output_path.read_text(encoding='utf-8'))
  assert saved['missingChecks'] == ['stop']
  updated = report_path.read_text(encoding='utf-8')
  assert 'PARTIAL' in updated
  assert updated.count('PENDING — fill in after listening') == 2


def test_apply_verdict_dry_run_does_not_write(tmp_path: Path) -> None:
  report_path = tmp_path / 'report.md'
  original = _sample_report()
  report_path.write_text(original, encoding='utf-8')
  output_path = tmp_path / 'verdict.json'
  applied_at = datetime(2026, 6, 26, 19, 5, tzinfo=UTC)

  result = verdict_mod.apply_verdict(
    _sample_payload(),
    output_path=output_path,
    report_paths=(report_path,),
    applied_at=applied_at,
    dry_run=True,
  )

  assert output_path.exists() is False
  assert report_path.read_text(encoding='utf-8') == original
  assert result['serialized'].startswith('{')


def test_parse_args_defaults() -> None:
  args = verdict_mod.parse_args([])
  assert args.output == verdict_mod.DEFAULT_VERDICT_JSON
  assert verdict_mod.default_reports(args) == verdict_mod.DEFAULT_REPORTS
