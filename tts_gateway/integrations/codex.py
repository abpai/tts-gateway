"""Install and run the Codex speech integration."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path

from tts_gateway.integrations import lifecycle
from tts_gateway.integrations.common import queue_speech
from tts_gateway.integrations.lifecycle import HookHandlers

_TARGET = 'Codex'


def installed_handlers(executable: str) -> HookHandlers:
  """Return the Codex hook handlers owned by tts-gateway."""
  command = shlex.join([executable, 'integrate', '--codex-event'])
  handler = {
    'type': 'command',
    'command': command,
    'timeout': 5,
  }
  return {
    'UserPromptSubmit': handler,
    'Stop': handler,
  }


def install(
  hooks_path: Path,
  record_path: Path,
  executable: str,
  root: Path,
) -> bool:
  """Install the Codex lifecycle hooks."""
  return lifecycle.install(
    lifecycle.InstallRequest(
      config_path=hooks_path,
      record_path=record_path,
      executable=executable,
      root=root,
      target=_TARGET,
      handlers=installed_handlers(executable),
    )
  )


def uninstall(hooks_path: Path, record_path: Path) -> bool:
  """Uninstall the Codex lifecycle hooks."""
  return lifecycle.uninstall(hooks_path, record_path, _TARGET)


def status(hooks_path: Path, record_path: Path) -> str:
  """Return the Codex integration status."""
  return lifecycle.status(hooks_path, record_path, _TARGET)


def handle_event(
  payload: str,
  record_path: Path,
  root: Path,
  enqueue: Callable[[str, str, Path], None] = queue_speech,
) -> None:
  """Record Codex prompts and queue marked final responses."""
  lifecycle.handle_event(payload, record_path, root, _TARGET, enqueue)
