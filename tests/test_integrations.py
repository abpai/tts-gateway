from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tts_gateway.integrations import claude, codex
from tts_gateway.integrations.common import (
  IntegrationError,
  queue_speech,
)

_TTS = '/Users/test/.local/bin/tts'


def _codex_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
  root = tmp_path / 'state'
  return tmp_path / 'hooks.json', root / 'codex.json', root


def _claude_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
  root = tmp_path / 'state'
  return tmp_path / 'settings.json', root / 'claude.json', root


def _codex_payload(event: str, **values: object) -> str:
  return json.dumps(
    {
      'session_id': 'session-1',
      'hook_event_name': event,
      **values,
    }
  )


def _claude_payload(event: str, **values: object) -> str:
  return json.dumps(
    {
      'session_id': 'session-1',
      'hook_event_name': event,
      **values,
    }
  )


def test_codex_install_merges_and_restores_existing_hooks(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  original_settings = {
    'description': 'Existing user hooks',
    'hooks': {
      'Stop': [
        {
          'hooks': [
            {
              'type': 'command',
              'command': '/usr/local/bin/existing',
            }
          ]
        }
      ]
    },
  }
  original = json.dumps(original_settings, indent=2) + '\n'
  hooks.write_text(original)

  assert codex.install(hooks, record, _TTS, root) is True
  installed = json.loads(hooks.read_text())
  assert installed['hooks']['Stop'][0] == original_settings['hooks']['Stop'][0]
  assert len(installed['hooks']['Stop']) == 2
  assert len(installed['hooks']['UserPromptSubmit']) == 1
  assert codex.status(hooks, record) == 'installed'
  assert codex.install(hooks, record, _TTS, root) is False

  assert codex.uninstall(hooks, record) is True
  assert hooks.read_text() == original
  assert codex.status(hooks, record) == 'not installed'


def test_codex_install_without_existing_config_removes_it_on_uninstall(
  tmp_path: Path,
) -> None:
  hooks, record, root = _codex_paths(tmp_path)

  assert codex.install(hooks, record, _TTS, root) is True
  assert hooks.exists()

  assert codex.uninstall(hooks, record) is True
  assert not hooks.exists()


def test_codex_uninstall_preserves_later_unrelated_changes(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  hooks.write_text('{"description": "Existing hooks"}\n')
  codex.install(hooks, record, _TTS, root)
  updated = json.loads(hooks.read_text())
  updated['version'] = 1
  hooks.write_text(json.dumps(updated, indent=2) + '\n')

  codex.uninstall(hooks, record)

  assert json.loads(hooks.read_text()) == {
    'description': 'Existing hooks',
    'version': 1,
  }


def test_codex_uninstall_refuses_changed_owned_hook(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  codex.install(hooks, record, _TTS, root)
  updated = json.loads(hooks.read_text())
  updated['hooks']['Stop'][-1]['hooks'][0]['timeout'] = 10
  hooks.write_text(json.dumps(updated, indent=2) + '\n')

  with pytest.raises(IntegrationError, match='changed after installation'):
    codex.uninstall(hooks, record)


def test_codex_install_writes_a_private_backup(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  hooks.write_text('{"description": "Existing hooks"}\n')

  codex.install(hooks, record, _TTS, root)

  backups = list((root / 'backups').iterdir())
  assert len(backups) == 1
  assert backups[0].read_text() == '{"description": "Existing hooks"}\n'


def test_codex_prompt_and_stop_queue_once(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  codex.install(hooks, record, _TTS, root)
  queued: list[tuple[str, str, Path]] = []

  def enqueue(text: str, executable: str, state: Path) -> None:
    queued.append((text, executable, state))

  codex.handle_event(
    _codex_payload('UserPromptSubmit', prompt='Explain it {speak back}'),
    record,
    root,
    enqueue=enqueue,
  )
  codex.handle_event(
    _codex_payload('Stop', last_assistant_message='The final answer.'),
    record,
    root,
    enqueue=enqueue,
  )
  codex.handle_event(
    _codex_payload('Stop', last_assistant_message='Duplicate answer.'),
    record,
    root,
    enqueue=enqueue,
  )

  assert queued == [('The final answer.', _TTS, root)]


def test_codex_unmarked_prompt_clears_pending_speech(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  codex.install(hooks, record, _TTS, root)
  queued: list[str] = []
  codex.handle_event(
    _codex_payload('UserPromptSubmit', prompt='First {speak back}'),
    record,
    root,
  )
  codex.handle_event(
    _codex_payload('UserPromptSubmit', prompt='Second without marker'),
    record,
    root,
  )
  codex.handle_event(
    _codex_payload('Stop', last_assistant_message='Not spoken.'),
    record,
    root,
    enqueue=lambda text, _executable, _root: queued.append(text),
  )

  assert queued == []


def test_codex_stop_without_assistant_message_is_ignored(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  codex.install(hooks, record, _TTS, root)
  queued: list[str] = []
  codex.handle_event(
    _codex_payload('UserPromptSubmit', prompt='Explain it {speak back}'),
    record,
    root,
  )

  codex.handle_event(
    _codex_payload('Stop', last_assistant_message=None),
    record,
    root,
    enqueue=lambda text, _executable, _root: queued.append(text),
  )

  assert queued == []


def test_codex_status_reports_drift(tmp_path: Path) -> None:
  hooks, record, root = _codex_paths(tmp_path)
  codex.install(hooks, record, _TTS, root)
  hooks.write_text('{}\n')

  assert codex.status(hooks, record) == 'drifted'


def test_claude_install_merges_and_restores_existing_hooks(tmp_path: Path) -> None:
  settings, record, root = _claude_paths(tmp_path)
  original_settings = {
    'hooks': {
      'Stop': [
        {
          'hooks': [
            {
              'type': 'command',
              'command': '/usr/local/bin/existing',
            }
          ]
        }
      ]
    },
    'permissions': {'defaultMode': 'auto'},
  }
  original = json.dumps(original_settings, indent=2) + '\n'
  settings.write_text(original)

  assert claude.install(settings, record, _TTS, root) is True
  installed = json.loads(settings.read_text())
  assert installed['hooks']['Stop'][0] == original_settings['hooks']['Stop'][0]
  assert len(installed['hooks']['Stop']) == 2
  assert len(installed['hooks']['UserPromptSubmit']) == 1
  assert claude.status(settings, record) == 'installed'
  assert claude.install(settings, record, _TTS, root) is False

  assert claude.uninstall(settings, record) is True
  assert settings.read_text() == original


def test_claude_install_without_existing_settings_removes_it_on_uninstall(
  tmp_path: Path,
) -> None:
  settings, record, root = _claude_paths(tmp_path)

  assert claude.install(settings, record, _TTS, root) is True
  assert settings.exists()

  assert claude.uninstall(settings, record) is True
  assert not settings.exists()


def test_claude_uninstall_preserves_later_unrelated_changes(tmp_path: Path) -> None:
  settings, record, root = _claude_paths(tmp_path)
  settings.write_text('{"permissions": {"defaultMode": "auto"}}\n')
  claude.install(settings, record, _TTS, root)
  updated = json.loads(settings.read_text())
  updated['model'] = 'sonnet'
  settings.write_text(json.dumps(updated, indent=2) + '\n')

  claude.uninstall(settings, record)

  final = json.loads(settings.read_text())
  assert final == {
    'permissions': {'defaultMode': 'auto'},
    'model': 'sonnet',
  }


def test_claude_uninstall_refuses_changed_owned_hook(tmp_path: Path) -> None:
  settings, record, root = _claude_paths(tmp_path)
  claude.install(settings, record, _TTS, root)
  updated = json.loads(settings.read_text())
  updated['hooks']['Stop'][-1]['hooks'][0]['timeout'] = 10
  settings.write_text(json.dumps(updated, indent=2) + '\n')

  with pytest.raises(IntegrationError, match='changed after installation'):
    claude.uninstall(settings, record)


def test_claude_prompt_and_stop_queue_once(tmp_path: Path) -> None:
  settings, record, root = _claude_paths(tmp_path)
  claude.install(settings, record, _TTS, root)
  queued: list[tuple[str, str, Path]] = []

  def enqueue(text: str, executable: str, state: Path) -> None:
    queued.append((text, executable, state))

  claude.handle_event(
    _claude_payload('UserPromptSubmit', prompt='Explain it {speak back}'),
    record,
    root,
    enqueue=enqueue,
  )
  claude.handle_event(
    _claude_payload('Stop', last_assistant_message='The final answer.'),
    record,
    root,
    enqueue=enqueue,
  )
  claude.handle_event(
    _claude_payload('Stop', last_assistant_message='Duplicate answer.'),
    record,
    root,
    enqueue=enqueue,
  )

  assert queued == [('The final answer.', _TTS, root)]


def test_claude_unmarked_prompt_clears_pending_speech(tmp_path: Path) -> None:
  settings, record, root = _claude_paths(tmp_path)
  claude.install(settings, record, _TTS, root)
  queued: list[str] = []
  claude.handle_event(
    _claude_payload('UserPromptSubmit', prompt='First {speak back}'),
    record,
    root,
  )
  claude.handle_event(
    _claude_payload('UserPromptSubmit', prompt='Second without marker'),
    record,
    root,
  )
  claude.handle_event(
    _claude_payload('Stop', last_assistant_message='Not spoken.'),
    record,
    root,
    enqueue=lambda text, _executable, _root: queued.append(text),
  )

  assert queued == []


def test_claude_status_reports_drift(tmp_path: Path) -> None:
  settings, record, root = _claude_paths(tmp_path)
  claude.install(settings, record, _TTS, root)
  settings.write_text('{}\n')

  assert claude.status(settings, record) == 'drifted'


def test_queue_speech_uses_private_file_and_detached_worker(tmp_path: Path) -> None:
  calls: list[tuple[list[str], dict[str, object]]] = []

  class FakeProcess:
    pass

  def spawn(
    command: list[str],
    **kwargs: object,
  ) -> Any:
    calls.append((command, kwargs))
    return FakeProcess()

  queue_speech('The final answer.', _TTS, tmp_path, spawn=spawn)

  command, kwargs = calls[0]
  speech_path = Path(command[-1])
  assert command == [_TTS, 'integrate', '--speak-file', str(speech_path)]
  assert speech_path.read_text() == 'The final answer.'
  assert speech_path.stat().st_mode & 0o777 == 0o600
  assert kwargs['start_new_session'] is True
