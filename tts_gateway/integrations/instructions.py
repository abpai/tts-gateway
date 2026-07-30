"""Install speak-back marker guidance in agent instruction files."""

from __future__ import annotations

from pathlib import Path

from tts_gateway.integrations.common import (
  IntegrationError,
  atomic_write_text,
  read_text,
  remove_config,
  speak_marker,
)

_BEGIN = '<!-- tts-gateway:begin -->'
_END = '<!-- tts-gateway:end -->'


def render_block() -> str:
  """Return the managed marker guidance block."""
  marker = speak_marker()
  return (
    f'{_BEGIN}\n'
    '## tts-gateway speak-back\n'
    '\n'
    f'The `{marker}` marker in a user prompt triggers a local text-to-speech\n'
    'hook that reads the final response aloud. The marker is not part of the\n'
    'request. Ignore it: do not act on it, repeat it, or mention it.\n'
    f'{_END}\n'
  )


def install(path: Path) -> bool:
  """Append the marker guidance block when it is missing."""
  _, text = read_text(path)
  if _BEGIN in text:
    return False
  atomic_write_text(path, text + _separator(text) + render_block())
  return True


def uninstall(path: Path) -> bool:
  """Remove the marker guidance block when it is present."""
  exists, text = read_text(path)
  if not exists or _BEGIN not in text:
    return False
  updated = _strip_block(path, text)
  if updated:
    atomic_write_text(path, updated)
  else:
    remove_config(path)
  return True


def _separator(text: str) -> str:
  if not text:
    return ''
  return '\n' if text.endswith('\n') else '\n\n'


def _strip_block(path: Path, text: str) -> str:
  start = text.index(_BEGIN)
  end = text.find(_END, start)
  if end < 0:
    raise IntegrationError(f'marker guidance block in {path} has no end marker')
  head = text[:start].rstrip('\n')
  tail = text[end + len(_END) :].lstrip('\n')
  parts = [part for part in (head, tail) if part]
  if not parts:
    return ''
  return '\n\n'.join(parts).rstrip('\n') + '\n'
