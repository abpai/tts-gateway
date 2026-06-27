#!/usr/bin/env python3
"""Apply explicit human listening verdict JSON to goal artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_CHECKS = (
  'clicks',
  'gaps',
  'prompt',
  'stop',
  'option-r',
  'kokoro-default',
)

CHECK_LABELS = {
  'clicks': 'No clicks or pops at chunk boundaries',
  'gaps': 'No audible gaps or dropped audio between chunks',
  'prompt': 'Playback starts promptly for the selected-text workflow',
  'stop': 'Stop Audio cancels playback during streaming',
  'option-r': 'Option+R replay on selected text sounds correct',
  'kokoro-default': 'Kokoro should remain the local latency default',
}

DEFAULT_VERDICT_JSON = Path('goals/tts-streaming-latency/listening/verdict.json')
DEFAULT_REPORTS = (
  Path('goals/tts-streaming-latency/listening/kokoro-current-report.md'),
  Path('goals/tts-streaming-latency/listening/cosyvoice3-zero-shot-m1-report.md'),
)

RECORDED_SECTION_HEADER = '## Recorded human verdict'
OVERALL_STATUS_PATTERN = re.compile(
  r'(## Overall human verdict\s*\n\s*\*\*Status:\*\* ).*?(?=\n\n|\Z)',
  re.DOTALL,
)
ENDPOINT_STATUS_PATTERN = re.compile(
  r'(### Human listening verdict\s*\n\s*\*\*Status:\*\* ).*?(?=\n\n|\Z)',
  re.DOTALL,
)
RECORDED_SECTION_PATTERN = re.compile(
  rf'\n{re.escape(RECORDED_SECTION_HEADER)}[\s\S]*\Z'
)


class VerdictValidationError(ValueError):
  """Raised when verdict input fails validation."""


class ReportUpdateError(ValueError):
  """Raised when a listening report cannot be updated safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--input',
    type=Path,
    help='Verdict JSON file (default: stdin)',
  )
  parser.add_argument(
    '--output',
    type=Path,
    default=DEFAULT_VERDICT_JSON,
    help=f'Output verdict JSON (default: {DEFAULT_VERDICT_JSON})',
  )
  parser.add_argument(
    '--report',
    action='append',
    dest='reports',
    type=Path,
    help='Markdown report to update (default: both listening reports)',
  )
  parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Print planned changes without writing files',
  )
  return parser.parse_args(argv)


def default_reports(args: argparse.Namespace) -> tuple[Path, ...]:
  return tuple(args.reports or DEFAULT_REPORTS)


def read_input_json(input_path: Path | None) -> dict[str, Any]:
  raw = input_path.read_text(encoding='utf-8') if input_path else sys.stdin.read()
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise VerdictValidationError(f'Invalid JSON: {exc}') from exc
  if not isinstance(payload, dict):
    raise VerdictValidationError('Verdict JSON must be an object')
  return payload


def validate_verdict_input(payload: dict[str, Any]) -> None:
  reviewed_at = payload.get('reviewedAt')
  if not isinstance(reviewed_at, str) or not reviewed_at.strip():
    raise VerdictValidationError('reviewedAt is required and must be nonblank')

  missing_keys = [key for key in REQUIRED_CHECKS if key not in payload]
  if missing_keys:
    joined = ', '.join(missing_keys)
    raise VerdictValidationError(f'Missing required boolean checks: {joined}')

  invalid_types = [
    key for key in REQUIRED_CHECKS if not isinstance(payload.get(key), bool)
  ]
  if invalid_types:
    joined = ', '.join(invalid_types)
    raise VerdictValidationError(f'Required checks must be booleans: {joined}')


def build_output_verdict(
  payload: dict[str, Any],
  *,
  applied_at: datetime,
) -> dict[str, Any]:
  missing_checks = [key for key in REQUIRED_CHECKS if not payload[key]]
  complete = not missing_checks
  status = 'PASS' if complete else 'PARTIAL'
  output = dict(payload)
  output['complete'] = complete
  output['status'] = status
  output['missingChecks'] = missing_checks
  output['appliedAt'] = applied_at.isoformat().replace('+00:00', 'Z')
  return output


def format_missing_checks(missing_checks: list[str]) -> str:
  labels = [CHECK_LABELS.get(key, key) for key in missing_checks]
  return ', '.join(labels)


def overall_status_line(status: str, missing_checks: list[str]) -> str:
  if status == 'PASS':
    return 'PASS — human listening checklist complete'
  detail = format_missing_checks(missing_checks)
  return f'PARTIAL — missing checks: {detail}'


def endpoint_status_line(status: str) -> str:
  if status == 'PASS':
    return 'PASS — human listening checklist complete'
  return 'PENDING — fill in after listening'


def render_recorded_section(verdict: dict[str, Any]) -> str:
  lines = [
    RECORDED_SECTION_HEADER,
    '',
    f'- reviewedAt: `{verdict["reviewedAt"]}`',
    f'- appliedAt: `{verdict["appliedAt"]}`',
    f'- status: {verdict["status"]}',
  ]
  notes = verdict.get('notes')
  if isinstance(notes, str) and notes.strip():
    lines.append(f'- notes: {notes.strip()}')
  lines.append('')
  return '\n'.join(lines)


def strip_recorded_section(content: str) -> str:
  return RECORDED_SECTION_PATTERN.sub('', content.rstrip())


def update_report_content(content: str, verdict: dict[str, Any]) -> str:
  updated = strip_recorded_section(content.rstrip())
  missing_checks = verdict['missingChecks']
  status = verdict['status']

  updated, overall_count = OVERALL_STATUS_PATTERN.subn(
    rf'\1{overall_status_line(status, missing_checks)}',
    updated,
    count=1,
  )
  if overall_count != 1:
    raise ReportUpdateError('expected exactly one overall verdict status')

  if status == 'PASS':
    endpoint_line = endpoint_status_line('PASS')
    endpoint_count = updated.count('### Human listening verdict')
    updated, changed_count = ENDPOINT_STATUS_PATTERN.subn(
      rf'\1{endpoint_line}',
      updated,
    )
    if endpoint_count == 0 or changed_count != endpoint_count:
      raise ReportUpdateError(
        'expected every endpoint listening verdict to include a status'
      )

  return f'{updated.rstrip()}\n\n{render_recorded_section(verdict)}'


def apply_verdict(
  payload: dict[str, Any],
  *,
  output_path: Path,
  report_paths: tuple[Path, ...],
  applied_at: datetime | None = None,
  dry_run: bool = False,
) -> dict[str, Any]:
  validate_verdict_input(payload)
  applied = applied_at or datetime.now(tz=UTC)
  verdict = build_output_verdict(payload, applied_at=applied)
  serialized = json.dumps(verdict, indent=2) + '\n'
  report_updates = {
    path: update_report_content(path.read_text(encoding='utf-8'), verdict)
    for path in report_paths
  }

  if dry_run:
    return {
      'verdict': verdict,
      'output_path': output_path,
      'report_paths': report_paths,
      'report_updates': report_updates,
      'serialized': serialized,
    }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(serialized, encoding='utf-8')
  for path, content in report_updates.items():
    path.write_text(content, encoding='utf-8')

  return {
    'verdict': verdict,
    'output_path': output_path,
    'report_paths': report_paths,
  }


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    payload = read_input_json(args.input)
    result = apply_verdict(
      payload,
      output_path=args.output,
      report_paths=default_reports(args),
      dry_run=args.dry_run,
    )
  except VerdictValidationError as exc:
    print(f'error: {exc}', file=sys.stderr)
    return 1
  except ReportUpdateError as exc:
    print(f'error: {exc}', file=sys.stderr)
    return 1
  except OSError as exc:
    print(f'error: {exc}', file=sys.stderr)
    return 1

  verdict = result['verdict']
  if args.dry_run:
    print(result['serialized'], end='')
    for path in result['report_paths']:
      print(f'would update {path}', file=sys.stderr)
    return 0

  print(f'Wrote {result["output_path"]} ({verdict["status"]})')
  for path in result['report_paths']:
    print(f'Updated {path}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
