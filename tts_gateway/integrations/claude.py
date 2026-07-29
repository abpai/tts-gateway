"""Install and run the Claude Code speech integration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tts_gateway.integrations import lifecycle
from tts_gateway.integrations.common import queue_speech
from tts_gateway.integrations.lifecycle import HookHandlers

_TARGET = 'Claude'
_EVENT_ARGS = ['integrate', '--claude-event']


def installed_handlers(executable: str) -> HookHandlers:
  """Return the Claude hook handlers owned by tts-gateway."""
  base: dict[str, Any] = {
    'type': 'command',
    'command': executable,
    'args': _EVENT_ARGS,
    'timeout': 5,
  }
  return {
    'UserPromptSubmit': base,
    'Stop': base,
  }


def install(
  settings_path: Path,
  record_path: Path,
  executable: str,
  root: Path,
) -> bool:
  """Install the Claude Code hooks."""
  return lifecycle.install(
    lifecycle.InstallRequest(
      config_path=settings_path,
      record_path=record_path,
      executable=executable,
      root=root,
      target=_TARGET,
      handlers=installed_handlers(executable),
    )
  )


def uninstall(settings_path: Path, record_path: Path) -> bool:
  """Uninstall the Claude Code hooks."""
  return lifecycle.uninstall(settings_path, record_path, _TARGET)


def status(settings_path: Path, record_path: Path) -> str:
  """Return the Claude integration status."""
  return lifecycle.status(settings_path, record_path, _TARGET)


def handle_event(
  payload: str,
  record_path: Path,
  root: Path,
  enqueue: Callable[[str, str, Path], None] = queue_speech,
) -> None:
  """Record Claude prompts and queue marked final responses."""
  lifecycle.handle_event(payload, record_path, root, _TARGET, enqueue)
