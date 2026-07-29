"""Shared coding-agent integration support."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, TextIO

from pydantic import BaseModel

from tts_gateway.speech_cli import SpeakOptions, SpeechCliError, speak

SPEAK_BACK_MARKER = '{speak back}'


def speak_marker() -> str:
  """Return the active speak-back marker (override via TTS_SPEAK_MARKER)."""
  return os.getenv('TTS_SPEAK_MARKER') or SPEAK_BACK_MARKER


class IntegrationError(SpeechCliError):
  """Report an agent integration failure."""


class InstallRecord(BaseModel):
  """Describe one installed coding-agent integration."""

  version: int = 1
  config_path: str
  tts_executable: str
  original_exists: bool
  original_text: str
  installed_hash: str
  backup_path: str


def state_root() -> Path:
  """Return the integration state directory."""
  configured = os.getenv('XDG_STATE_HOME')
  base = Path(configured).expanduser() if configured else Path.home() / '.local/state'
  return base / 'tts-gateway' / 'integrations'


def codex_hooks_path() -> Path:
  """Return the active user Codex hooks path."""
  codex_home = os.getenv('CODEX_HOME')
  base = Path(codex_home).expanduser() if codex_home else Path.home() / '.codex'
  return base / 'hooks.json'


def claude_settings_path() -> Path:
  """Return the active user Claude settings path."""
  config_dir = os.getenv('CLAUDE_CONFIG_DIR')
  base = Path(config_dir).expanduser() if config_dir else Path.home() / '.claude'
  return base / 'settings.json'


def resolve_tts_executable() -> str:
  """Return the stable installed tts executable."""
  executable = shutil.which('tts')
  if executable is None:
    raise IntegrationError('tts executable not found on PATH')
  return str(Path(executable).expanduser().absolute())


def text_hash(text: str) -> str:
  """Return a stable text hash."""
  return hashlib.sha256(text.encode()).hexdigest()


def read_text(path: Path) -> tuple[bool, str]:
  """Read a file and report whether it exists."""
  if not path.exists():
    return False, ''
  try:
    return True, path.read_text(encoding='utf-8')
  except OSError as exc:
    raise IntegrationError(f'cannot read {path}: {exc}') from exc


def atomic_write_text(path: Path, text: str) -> None:
  """Replace a text file atomically."""
  descriptor, temporary, mode = _create_temporary_file(path)
  temporary_path = Path(temporary)
  try:
    os.fchmod(descriptor, mode)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
      output.write(text)
    descriptor = -1
    os.replace(temporary_path, path)
  except OSError as exc:
    if descriptor >= 0:
      _close_descriptor(descriptor)
    temporary_path.unlink(missing_ok=True)
    raise IntegrationError(f'cannot write {path}: {exc}') from exc


def _create_temporary_file(path: Path) -> tuple[int, str, int]:
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    descriptor, temporary = tempfile.mkstemp(
      dir=path.parent,
      prefix=f'.{path.name}.',
    )
    return descriptor, temporary, mode
  except OSError as exc:
    raise IntegrationError(f'cannot prepare {path}: {exc}') from exc


def _close_descriptor(descriptor: int) -> None:
  try:
    os.close(descriptor)
  except OSError:
    pass


def write_backup(path: Path, text: str, target: str, root: Path) -> Path:
  """Write a dated configuration backup."""
  timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
  backup = root / 'backups' / f'{target}-{timestamp}{path.suffix}'
  atomic_write_text(backup, text)
  return backup


def write_record(path: Path, record: BaseModel) -> None:
  """Write an integration record."""
  atomic_write_text(path, record.model_dump_json(indent=2) + '\n')


def read_record(path: Path, model: type[BaseModel]) -> BaseModel | None:
  """Read and validate an integration record."""
  if not path.exists():
    return None
  try:
    return model.model_validate_json(path.read_text(encoding='utf-8'))
  except (OSError, ValueError) as exc:
    raise IntegrationError(f'invalid integration state at {path}: {exc}') from exc


def remove_record(path: Path) -> None:
  """Remove one integration record."""
  try:
    path.unlink(missing_ok=True)
  except OSError as exc:
    raise IntegrationError(f'cannot remove integration state at {path}: {exc}') from exc


def remove_config(path: Path) -> None:
  """Remove a configuration file created by the installer."""
  try:
    path.unlink(missing_ok=True)
  except OSError as exc:
    raise IntegrationError(f'cannot remove {path}: {exc}') from exc


def read_event_payload(stdin: TextIO) -> str:
  """Read one event payload from standard input."""
  payload = stdin.read()
  if not payload.strip():
    raise IntegrationError('integration event payload is required')
  return payload


def speak_text(text: str, stdout: BinaryIO) -> None:
  """Speak text through the configured local gateway."""
  options = SpeakOptions(
    text=text,
    base_url=os.getenv('TTS_GATEWAY_URL', 'http://127.0.0.1:45123'),
  )
  speak(options, stdout)


def queue_speech(
  text: str,
  executable: str,
  root: Path,
  spawn: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> None:
  """Start detached speech playback from a private text file."""
  descriptor, speech_path = _create_speech_file(root)
  try:
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
      output.write(text)
    descriptor = -1
    _spawn_speech(executable, speech_path, spawn)
  except OSError as exc:
    if descriptor >= 0:
      _close_descriptor(descriptor)
    speech_path.unlink(missing_ok=True)
    raise IntegrationError(f'cannot queue speech: {exc}') from exc


def _create_speech_file(root: Path) -> tuple[int, Path]:
  queue = root / 'queue'
  try:
    queue.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
      dir=queue,
      prefix='speech-',
      suffix='.txt',
    )
  except OSError as exc:
    raise IntegrationError(f'cannot create the speech queue: {exc}') from exc
  return descriptor, Path(temporary)


def _spawn_speech(
  executable: str,
  speech_path: Path,
  spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
  spawn(
    [executable, 'integrate', '--speak-file', str(speech_path)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
  )


def speak_file(path: Path, stdout: BinaryIO) -> None:
  """Speak and remove one queued text file."""
  try:
    text = path.read_text(encoding='utf-8')
  except OSError as exc:
    raise IntegrationError(f'cannot read queued speech at {path}: {exc}') from exc
  finally:
    path.unlink(missing_ok=True)
  if text.strip():
    speak_text(text, stdout)
