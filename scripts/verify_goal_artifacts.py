#!/usr/bin/env python3
"""Verify goal artifact completeness for streaming latency work."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

AuditLevel = Literal['PASS', 'WARN', 'FAIL']

DEFAULT_GOAL_DIR = Path('goals/tts-streaming-latency')
REQUIRED_DOCS = ('goal.md', 'facts.md', 'plan.md')
REQUIRED_BENCHMARKS = (
  'benchmarks/kokoro-baseline.json',
  'benchmarks/stream-transport-comparison.json',
  'benchmarks/kokoro-current-engine-metadata.json',
  'benchmarks/cosyvoice3-zero-shot-m1-short.json',
  'benchmarks/cosyvoice3-zero-shot-m1-sentence.json',
)
KOKORO_CURRENT_BENCHMARK = 'benchmarks/kokoro-current-engine-metadata.json'
COSYVOICE_BENCHMARKS = (
  'benchmarks/cosyvoice3-zero-shot-m1-short.json',
  'benchmarks/cosyvoice3-zero-shot-m1-sentence.json',
)
PCM_BENCHMARKS = (
  'benchmarks/stream-transport-comparison.json',
  KOKORO_CURRENT_BENCHMARK,
  *COSYVOICE_BENCHMARKS,
)
LISTENING_REPORTS = (
  'listening/kokoro-current-report.md',
  'listening/cosyvoice3-zero-shot-m1-report.md',
)
PCM_STREAM = '/tts/stream/pcm'
KOKORO_BASELINE_COMPARE_SUFFIX = 'kokoro-current-engine-metadata.json'
HUMAN_VERDICT_HEADER = '### Human listening verdict'
OVERALL_VERDICT_HEADER = '## Overall human verdict'
AUDIO_SRC_PATTERN = re.compile(r'<audio[^>]+src="([^"]+)"', re.IGNORECASE)
WAVEFORM_SECTION_PATTERN = re.compile(
  r'### Waveform sanity\s*\n(.*?)(?=\n### |\n## |\Z)',
  re.DOTALL,
)


class AuditFinding(BaseModel):
  """Single audit check result."""

  level: AuditLevel
  check: str
  detail: str = ''


class BenchmarkShape(BaseModel):
  """Required benchmark JSON fields for goal verification."""

  generated_at: str = Field(alias='generatedAt')
  stream_endpoints: list[str] = Field(alias='streamEndpoints')
  fixtures: list[Any]
  runs: list[Any]
  summary: list[Any]
  engine: str | None = None
  compare_baseline: str | None = Field(default=None, alias='compareBaseline')
  health: dict[str, Any] | None = None


def endpoint_values(items: list[Any]) -> set[str]:
  values: set[str] = set()
  for item in items:
    if isinstance(item, dict) and isinstance(item.get('endpoint'), str):
      values.add(item['endpoint'])
  return values


def _load_apply_listening_verdict():
  module_name = '_tts_gateway_apply_listening_verdict'
  if module_name in sys.modules:
    return sys.modules[module_name]
  script_path = Path(__file__).resolve().parent / 'apply_listening_verdict.py'
  spec = importlib.util.spec_from_file_location(module_name, script_path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--goal-dir',
    type=Path,
    default=DEFAULT_GOAL_DIR,
    help=f'Goal directory to verify (default: {DEFAULT_GOAL_DIR})',
  )
  parser.add_argument(
    '--require-human-verdict',
    action='store_true',
    help='Fail when listening/verdict.json is missing or incomplete',
  )
  return parser.parse_args(argv)


def record(
  findings: list[AuditFinding],
  *,
  level: AuditLevel,
  check: str,
  detail: str = '',
) -> None:
  findings.append(AuditFinding(level=level, check=check, detail=detail))


def check_required_docs(goal_dir: Path, findings: list[AuditFinding]) -> None:
  for name in REQUIRED_DOCS:
    path = goal_dir / name
    if path.is_file():
      record(findings, level='PASS', check=f'doc:{name}')
    else:
      record(findings, level='FAIL', check=f'doc:{name}', detail='missing')


def load_benchmark(path: Path) -> tuple[BenchmarkShape | None, str | None]:
  try:
    payload = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as exc:
    return None, f'invalid JSON: {exc}'
  if not isinstance(payload, dict):
    return None, 'benchmark root must be an object'
  try:
    return BenchmarkShape.model_validate(payload), None
  except ValidationError as exc:
    return None, str(exc)


def check_benchmark_common(
  rel_path: str,
  report: BenchmarkShape,
  findings: list[AuditFinding],
) -> None:
  prefix = f'benchmark:{rel_path}'
  if not report.generated_at.strip():
    record(findings, level='FAIL', check=prefix, detail='generatedAt is blank')
    return
  if not report.fixtures:
    record(findings, level='FAIL', check=prefix, detail='fixtures is empty')
    return
  if not report.runs:
    record(findings, level='FAIL', check=prefix, detail='runs is empty')
    return
  if not report.summary:
    record(findings, level='FAIL', check=prefix, detail='summary is empty')
    return
  if not report.stream_endpoints:
    record(findings, level='FAIL', check=prefix, detail='streamEndpoints is empty')
    return
  record(findings, level='PASS', check=prefix)


def check_pcm_endpoints(
  rel_path: str,
  report: BenchmarkShape,
  findings: list[AuditFinding],
) -> None:
  if PCM_STREAM not in report.stream_endpoints:
    record(
      findings,
      level='FAIL',
      check=f'benchmark:{rel_path}:pcm',
      detail=f'missing stream endpoint {PCM_STREAM}',
    )
    return
  if PCM_STREAM not in endpoint_values(report.runs):
    record(
      findings,
      level='FAIL',
      check=f'benchmark:{rel_path}:pcm',
      detail=f'runs missing endpoint {PCM_STREAM}',
    )
    return
  if PCM_STREAM not in endpoint_values(report.summary):
    record(
      findings,
      level='FAIL',
      check=f'benchmark:{rel_path}:pcm',
      detail=f'summary missing endpoint {PCM_STREAM}',
    )
    return
  record(findings, level='PASS', check=f'benchmark:{rel_path}:pcm')


def check_kokoro_current(
  report: BenchmarkShape,
  findings: list[AuditFinding],
) -> None:
  rel_path = KOKORO_CURRENT_BENCHMARK
  health = report.health or {}
  primary = health.get('primaryEngine')
  if primary != 'kokoro':
    record(
      findings,
      level='FAIL',
      check=f'benchmark:{rel_path}:health',
      detail=f'primaryEngine expected kokoro, got {primary!r}',
    )
    return
  record(findings, level='PASS', check=f'benchmark:{rel_path}:health')


def check_cosyvoice_benchmark(
  rel_path: str,
  report: BenchmarkShape,
  findings: list[AuditFinding],
) -> None:
  if report.engine != 'cosyvoice':
    record(
      findings,
      level='FAIL',
      check=f'benchmark:{rel_path}:engine',
      detail=f'engine expected cosyvoice, got {report.engine!r}',
    )
    return
  record(findings, level='PASS', check=f'benchmark:{rel_path}:engine')

  baseline = report.compare_baseline or ''
  if KOKORO_BASELINE_COMPARE_SUFFIX not in baseline:
    record(
      findings,
      level='FAIL',
      check=f'benchmark:{rel_path}:compare',
      detail='compareBaseline must reference kokoro-current-engine-metadata.json',
    )
    return
  record(findings, level='PASS', check=f'benchmark:{rel_path}:compare')


def check_benchmarks(goal_dir: Path, findings: list[AuditFinding]) -> None:
  for rel_path in REQUIRED_BENCHMARKS:
    path = goal_dir / rel_path
    if not path.is_file():
      record(findings, level='FAIL', check=f'benchmark:{rel_path}', detail='missing')
      continue

    report, error = load_benchmark(path)
    if report is None:
      record(
        findings,
        level='FAIL',
        check=f'benchmark:{rel_path}',
        detail=error or 'invalid benchmark',
      )
      continue

    check_benchmark_common(rel_path, report, findings)
    if rel_path in PCM_BENCHMARKS:
      check_pcm_endpoints(rel_path, report, findings)
    if rel_path == KOKORO_CURRENT_BENCHMARK:
      check_kokoro_current(report, findings)
    if rel_path in COSYVOICE_BENCHMARKS:
      check_cosyvoice_benchmark(rel_path, report, findings)


def waveform_sections_pass(content: str) -> tuple[bool, str]:
  sections = WAVEFORM_SECTION_PATTERN.findall(content)
  if not sections:
    return False, 'no waveform sanity sections found'
  for index, section in enumerate(sections, start=1):
    if '- status: PASS' not in section:
      status_match = re.search(r'- status: (\w+)', section)
      status = status_match.group(1) if status_match else 'missing'
      return False, f'waveform section {index} status is {status}'
  return True, ''


def check_listening_report(
  rel_path: str,
  content: str,
  findings: list[AuditFinding],
) -> None:
  prefix = f'listening:{rel_path}'
  ok, detail = waveform_sections_pass(content)
  if not ok:
    record(findings, level='FAIL', check=f'{prefix}:waveform', detail=detail)
  else:
    record(findings, level='PASS', check=f'{prefix}:waveform')

  if HUMAN_VERDICT_HEADER not in content:
    record(
      findings,
      level='FAIL',
      check=f'{prefix}:human-section',
      detail='missing per-endpoint human verdict section',
    )
  elif OVERALL_VERDICT_HEADER not in content:
    record(
      findings,
      level='FAIL',
      check=f'{prefix}:human-section',
      detail='missing overall human verdict section',
    )
  else:
    record(findings, level='PASS', check=f'{prefix}:human-section')


def check_listening_reports(goal_dir: Path, findings: list[AuditFinding]) -> None:
  for rel_path in LISTENING_REPORTS:
    path = goal_dir / rel_path
    if not path.is_file():
      record(findings, level='FAIL', check=f'listening:{rel_path}', detail='missing')
      continue
    check_listening_report(rel_path, path.read_text(encoding='utf-8'), findings)


def check_review_html(goal_dir: Path, findings: list[AuditFinding]) -> None:
  rel_path = Path('listening/review.html')
  path = goal_dir / rel_path
  if not path.is_file():
    record(findings, level='FAIL', check='listening:review.html', detail='missing')
    return

  content = path.read_text(encoding='utf-8')
  sources = AUDIO_SRC_PATTERN.findall(content)
  if not sources:
    record(
      findings,
      level='FAIL',
      check='listening:review.html:audio',
      detail='no audio src references found',
    )
    return

  listening_dir = goal_dir / 'listening'
  missing: list[str] = []
  for src in sources:
    asset_path = (listening_dir / src).resolve()
    try:
      asset_path.relative_to(listening_dir.resolve())
    except ValueError:
      missing.append(f'{src} (escapes listening dir)')
      continue
    if not asset_path.is_file():
      missing.append(src)

  if missing:
    record(
      findings,
      level='FAIL',
      check='listening:review.html:audio',
      detail=f'missing assets: {", ".join(missing)}',
    )
    return
  record(findings, level='PASS', check='listening:review.html:audio')


def verdict_is_complete(
  payload: dict[str, Any], required_checks: tuple[str, ...]
) -> bool:
  return all(payload.get(key) is True for key in required_checks)


def check_verdict(
  goal_dir: Path,
  findings: list[AuditFinding],
  *,
  require_human_verdict: bool,
) -> None:
  verdict_mod = _load_apply_listening_verdict()
  path = goal_dir / 'listening/verdict.json'
  if not path.is_file():
    level: AuditLevel = 'FAIL' if require_human_verdict else 'WARN'
    record(
      findings,
      level=level,
      check='listening:verdict.json',
      detail='missing human verdict',
    )
    return

  try:
    payload = verdict_mod.read_input_json(path)
  except verdict_mod.VerdictValidationError as exc:
    record(
      findings,
      level='FAIL',
      check='listening:verdict.json',
      detail=str(exc),
    )
    return

  try:
    verdict_mod.validate_verdict_input(payload)
  except verdict_mod.VerdictValidationError as exc:
    record(
      findings,
      level='FAIL',
      check='listening:verdict.json',
      detail=str(exc),
    )
    return

  record(findings, level='PASS', check='listening:verdict.json:schema')

  if verdict_is_complete(payload, verdict_mod.REQUIRED_CHECKS):
    record(findings, level='PASS', check='listening:verdict.json:complete')
    return

  missing = [key for key in verdict_mod.REQUIRED_CHECKS if not payload.get(key)]
  detail = f'incomplete checks: {", ".join(missing)}'
  record(
    findings,
    level='FAIL',
    check='listening:verdict.json:complete',
    detail=detail,
  )


def audit_exit_code(
  findings: list[AuditFinding],
  *,
  require_human_verdict: bool,
) -> int:
  has_fail = any(finding.level == 'FAIL' for finding in findings)
  if has_fail:
    return 1

  verdict_path_missing = any(
    finding.check == 'listening:verdict.json' and finding.level == 'WARN'
    for finding in findings
  )
  if verdict_path_missing and require_human_verdict:
    return 1
  return 0


def run_audit(
  goal_dir: Path,
  *,
  require_human_verdict: bool = False,
) -> tuple[list[AuditFinding], int]:
  findings: list[AuditFinding] = []
  if not goal_dir.is_dir():
    record(
      findings, level='FAIL', check='goal-dir', detail=f'not a directory: {goal_dir}'
    )
    return findings, 1

  check_required_docs(goal_dir, findings)
  check_benchmarks(goal_dir, findings)
  check_listening_reports(goal_dir, findings)
  check_review_html(goal_dir, findings)
  check_verdict(goal_dir, findings, require_human_verdict=require_human_verdict)
  return findings, audit_exit_code(
    findings,
    require_human_verdict=require_human_verdict,
  )


def format_findings(findings: list[AuditFinding]) -> str:
  lines: list[str] = []
  for finding in findings:
    lines.append(f'{finding.level:<4} {finding.check}')
    if finding.detail:
      lines.append(f'     {finding.detail}')
  return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  findings, exit_code = run_audit(
    args.goal_dir,
    require_human_verdict=args.require_human_verdict,
  )
  print(format_findings(findings))
  return exit_code


if __name__ == '__main__':
  raise SystemExit(main())
