#!/usr/bin/env python3
"""Prepare and push a tagged release."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Literal

VERSION_RE = re.compile(r'^\d+\.\d+\.\d+$')
PYPROJECT_PATH = Path('pyproject.toml')
BumpKind = Literal['patch', 'minor', 'major']


class ReleaseError(RuntimeError):
  """Release preparation failed."""


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Bump project version, verify, commit, tag, and push a release.'
  )
  parser.add_argument(
    'version',
    nargs='?',
    help='Release version, e.g. 0.1.3. If omitted, bump the current version.',
  )
  parser.add_argument(
    '--bump',
    choices=['patch', 'minor', 'major'],
    default='patch',
    help='Version segment to bump when version is omitted.',
  )
  parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Print planned changes and commands without modifying git state.',
  )
  parser.add_argument(
    '--skip-checks',
    action='store_true',
    help='Skip lint, typecheck, tests, and package validation.',
  )
  return parser.parse_args()


def validate_version(version: str) -> None:
  if not VERSION_RE.fullmatch(version):
    raise ReleaseError('version must look like X.Y.Z')


def run_command(command: list[str], *, dry_run: bool) -> None:
  print('$', ' '.join(command))
  if dry_run:
    return
  subprocess.run(command, check=True)


def capture_command(command: list[str]) -> str:
  result = subprocess.run(command, check=True, capture_output=True, text=True)
  return result.stdout.strip()


def ensure_clean_worktree() -> None:
  status = capture_command(['git', 'status', '--short'])
  if status:
    raise ReleaseError('git worktree must be clean before releasing')


def ensure_tag_missing(tag_name: str) -> None:
  tags = capture_command(['git', 'tag', '--list', tag_name])
  if tags:
    raise ReleaseError(f'git tag {tag_name!r} already exists')


def read_current_version() -> str:
  for line in PYPROJECT_PATH.read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith('version = '):
      return stripped.split('"')[1]
  raise ReleaseError('could not find project version in pyproject.toml')


def bump_version(version: str, bump: BumpKind) -> str:
  major, minor, patch = (int(part) for part in version.split('.'))
  if bump == 'major':
    return f'{major + 1}.0.0'
  if bump == 'minor':
    return f'{major}.{minor + 1}.0'
  return f'{major}.{minor}.{patch + 1}'


def update_version(version: str, *, dry_run: bool) -> None:
  lines = PYPROJECT_PATH.read_text().splitlines()
  updated_lines: list[str] = []
  in_project_section = False
  replaced = False

  for line in lines:
    stripped = line.strip()
    if stripped == '[project]':
      in_project_section = True
      updated_lines.append(line)
      continue
    if stripped.startswith('[') and stripped != '[project]':
      in_project_section = False
    if in_project_section and stripped.startswith('version = '):
      updated_lines.append(f'version = "{version}"')
      replaced = True
      continue
    updated_lines.append(line)

  if not replaced:
    raise ReleaseError('could not update project.version in pyproject.toml')

  print(f'Updating pyproject.toml to version {version}')
  if not dry_run:
    PYPROJECT_PATH.write_text('\n'.join(updated_lines) + '\n')


def run_checks(*, dry_run: bool) -> None:
  commands: list[list[str]] = [
    ['uv', 'run', 'ruff', 'check', '.'],
    ['uv', 'run', 'ty', 'check'],
    ['uv', 'run', 'pytest'],
    ['rm', '-rf', 'dist'],
    ['uv', 'build'],
  ]
  for command in commands:
    run_command(command, dry_run=dry_run)
  # twine needs real file paths — glob after build
  if dry_run:
    print('$ uvx twine check dist/*')
  else:
    dist_files = sorted(
      str(p) for p in Path('dist').iterdir() if p.suffix in ('.whl', '.gz')
    )
    if not dist_files:
      raise ReleaseError('uv build produced no artifacts in dist/')
    run_command(['uvx', 'twine', 'check', *dist_files], dry_run=False)


def commit_tag_and_push(version: str, *, dry_run: bool) -> None:
  tag_name = f'v{version}'
  run_command(['git', 'add', 'pyproject.toml', 'uv.lock'], dry_run=dry_run)
  run_command(
    ['git', 'commit', '-m', f'chore: release {tag_name}'],
    dry_run=dry_run,
  )
  run_command(['git', 'tag', tag_name], dry_run=dry_run)
  run_command(['git', 'push', 'origin', 'HEAD'], dry_run=dry_run)
  run_command(['git', 'push', 'origin', tag_name], dry_run=dry_run)


def main() -> int:
  args = parse_args()
  current_version = read_current_version()
  version = args.version or bump_version(current_version, args.bump)
  validate_version(version)
  ensure_clean_worktree()
  ensure_tag_missing(f'v{version}')

  if current_version == version:
    raise ReleaseError(f'project.version is already {version}')

  update_version(version, dry_run=args.dry_run)
  if not args.skip_checks:
    run_checks(dry_run=args.dry_run)
  commit_tag_and_push(version, dry_run=args.dry_run)
  return 0


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except ReleaseError as exc:
    print(f'error: {exc}')
    raise SystemExit(1) from exc
