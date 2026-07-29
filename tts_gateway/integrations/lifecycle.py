"""Install and run final-response lifecycle hooks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from tts_gateway.integrations.common import (
  SPEAK_BACK_MARKER,
  InstallRecord,
  IntegrationError,
  atomic_write_text,
  queue_speech,
  read_record,
  read_text,
  remove_config,
  remove_record,
  text_hash,
  write_backup,
  write_record,
)

HookHandlers = dict[str, dict[str, Any]]


class LifecycleInstallRecord(InstallRecord):
  """Describe one installed lifecycle-hook integration."""

  target: str
  installed_handlers: HookHandlers


class LifecycleEvent(BaseModel):
  """Describe a supported agent lifecycle-hook event."""

  model_config = ConfigDict(extra='ignore')

  session_id: str
  hook_event_name: str
  prompt: str = ''
  last_assistant_message: str | None = None


class InstallRequest(BaseModel):
  """Describe one lifecycle-hook installation."""

  model_config = ConfigDict(frozen=True)

  config_path: Path
  record_path: Path
  executable: str
  root: Path
  target: str
  handlers: HookHandlers


def install(request: InstallRequest) -> bool:
  """Install one lifecycle-hook integration."""
  existing = _load_record(request.record_path)
  exists, original = read_text(request.config_path)
  settings = _parse_settings(original, request.target)
  if existing is not None:
    return _confirm_existing_install(
      settings,
      request.handlers,
      request.target,
      existing,
    )
  updated, record = _prepare_install(request, exists, original, settings)
  _commit_install(request.config_path, updated, request.record_path, record)
  return True


def _prepare_install(
  request: InstallRequest,
  exists: bool,
  original: str,
  settings: dict[str, Any],
) -> tuple[str, LifecycleInstallRecord]:
  if _contains_any_handler(settings, request.handlers):
    message = f'{request.target} integration exists without installation state'
    raise IntegrationError(message)
  updated = _render_settings(_add_handlers(settings, request.handlers, request.target))
  backup = write_backup(
    request.config_path, original, request.target.lower(), request.root
  )
  record = _new_record(
    request.config_path,
    request.executable,
    exists,
    original,
    updated,
    backup,
    request.target,
    request.handlers,
  )
  return updated, record


def _new_record(
  config_path: Path,
  executable: str,
  exists: bool,
  original: str,
  updated: str,
  backup: Path,
  target: str,
  handlers: HookHandlers,
) -> LifecycleInstallRecord:
  return LifecycleInstallRecord(
    config_path=str(config_path),
    tts_executable=executable,
    original_exists=exists,
    original_text=original,
    installed_hash=text_hash(updated),
    backup_path=str(backup),
    target=target,
    installed_handlers=handlers,
  )


def _commit_install(
  config_path: Path,
  updated: str,
  record_path: Path,
  record: LifecycleInstallRecord,
) -> None:
  write_record(record_path, record)
  try:
    atomic_write_text(config_path, updated)
  except IntegrationError:
    remove_record(record_path)
    raise


def uninstall(config_path: Path, record_path: Path, target: str) -> bool:
  """Uninstall one lifecycle-hook integration."""
  record = _load_record(record_path)
  if record is None:
    return False
  _confirm_target(record, target)
  exists, current = read_text(config_path)
  settings = _parse_settings(current, target)
  if not exists or not _contains_all_handlers(settings, record.installed_handlers):
    raise IntegrationError(f'{target} hook configuration changed after installation')
  if text_hash(current) == record.installed_hash:
    _restore_original(config_path, record)
  else:
    updated = _remove_handlers(settings, record.installed_handlers, target)
    atomic_write_text(config_path, _render_settings(updated))
  remove_record(record_path)
  return True


def status(config_path: Path, record_path: Path, target: str) -> str:
  """Return one lifecycle-hook integration status."""
  record = _load_record(record_path)
  if record is None:
    return 'not installed'
  if record.target != target:
    return 'drifted'
  exists, current = read_text(config_path)
  if not exists:
    return 'drifted'
  settings = _parse_settings(current, target)
  return (
    'installed'
    if _contains_all_handlers(settings, record.installed_handlers)
    else 'drifted'
  )


def handle_event(
  payload: str,
  record_path: Path,
  root: Path,
  target: str,
  enqueue: Callable[[str, str, Path], None] = queue_speech,
) -> None:
  """Record prompts and queue marked final responses."""
  record = _load_record(record_path)
  if record is None:
    raise IntegrationError(f'{target} integration state is missing')
  _confirm_target(record, target)
  event = _parse_event(payload, target)
  flag = _flag_path(root, target, event.session_id)
  if event.hook_event_name == 'UserPromptSubmit':
    _set_prompt_flag(flag, event.prompt)
    return
  if event.hook_event_name != 'Stop' or not _claim_prompt_flag(flag, target):
    return
  message = event.last_assistant_message
  if message and message.strip():
    enqueue(message, record.tts_executable, root)


def _parse_settings(text: str, target: str) -> dict[str, Any]:
  if not text.strip():
    return {}
  try:
    settings = json.loads(text)
  except json.JSONDecodeError as exc:
    raise IntegrationError(f'invalid {target} hooks JSON: {exc}') from exc
  if not isinstance(settings, dict):
    raise IntegrationError(f'{target} hooks must contain a JSON object')
  return settings


def _render_settings(settings: dict[str, Any]) -> str:
  return json.dumps(settings, indent=2) + '\n'


def _add_handlers(
  settings: dict[str, Any],
  handlers: HookHandlers,
  target: str,
) -> dict[str, Any]:
  updated = _copy_settings(settings)
  hooks = updated.setdefault('hooks', {})
  if not isinstance(hooks, dict):
    raise IntegrationError(f'{target} hooks must contain a JSON object')
  for event, handler in handlers.items():
    groups = hooks.setdefault(event, [])
    if not isinstance(groups, list):
      raise IntegrationError(f'{target} {event} hooks must contain an array')
    groups.append({'hooks': [handler]})
  return updated


def _remove_handlers(
  settings: dict[str, Any],
  handlers: HookHandlers,
  target: str,
) -> dict[str, Any]:
  updated = _copy_settings(settings)
  hooks = updated.get('hooks')
  if not isinstance(hooks, dict):
    raise IntegrationError(f'{target} hooks must contain a JSON object')
  for event, handler in handlers.items():
    groups = hooks.get(event)
    if isinstance(groups, list):
      hooks[event] = _remove_event_handler(groups, handler)
      if not hooks[event]:
        hooks.pop(event)
  if not hooks:
    updated.pop('hooks', None)
  return updated


def _remove_event_handler(
  groups: list[object],
  owned: dict[str, Any],
) -> list[object]:
  updated: list[object] = []
  for group in groups:
    if not isinstance(group, dict):
      updated.append(group)
      continue
    group_dict = cast(dict[str, Any], group)
    raw_handlers = group_dict.get('hooks')
    if not isinstance(raw_handlers, list):
      updated.append(group)
      continue
    handlers = [item for item in raw_handlers if item != owned]
    if handlers:
      updated.append({**group_dict, 'hooks': handlers})
  return updated


def _contains_any_handler(
  settings: dict[str, Any],
  handlers: HookHandlers,
) -> bool:
  return any(
    _contains_handler(settings, event, handler) for event, handler in handlers.items()
  )


def _contains_all_handlers(
  settings: dict[str, Any],
  handlers: HookHandlers,
) -> bool:
  return all(
    _contains_handler(settings, event, handler) for event, handler in handlers.items()
  )


def _contains_handler(
  settings: dict[str, Any],
  event: str,
  handler: dict[str, Any],
) -> bool:
  hooks = settings.get('hooks')
  if not isinstance(hooks, dict):
    return False
  groups = hooks.get(event)
  if not isinstance(groups, list):
    return False
  return any(
    isinstance(group, dict)
    and isinstance(group.get('hooks'), list)
    and handler in group['hooks']
    for group in groups
  )


def _copy_settings(settings: dict[str, Any]) -> dict[str, Any]:
  return json.loads(json.dumps(settings))


def _confirm_existing_install(
  settings: dict[str, Any],
  handlers: HookHandlers,
  target: str,
  record: LifecycleInstallRecord,
) -> bool:
  _confirm_target(record, target)
  if record.installed_handlers != handlers:
    raise IntegrationError(f'{target} integration uses another tts executable')
  if not _contains_all_handlers(settings, handlers):
    raise IntegrationError(f'{target} integration state does not match its hooks')
  return False


def _confirm_target(record: LifecycleInstallRecord, target: str) -> None:
  if record.target != target:
    raise IntegrationError(f'{target} integration state has the wrong target')


def _load_record(path: Path) -> LifecycleInstallRecord | None:
  record = read_record(path, LifecycleInstallRecord)
  return record if isinstance(record, LifecycleInstallRecord) else None


def _parse_event(payload: str, target: str) -> LifecycleEvent:
  try:
    return LifecycleEvent.model_validate_json(payload)
  except ValueError as exc:
    raise IntegrationError(f'invalid {target} hook event: {exc}') from exc


def _flag_path(root: Path, target: str, session_id: str) -> Path:
  digest = hashlib.sha256(session_id.encode()).hexdigest()
  return root / f'{target.lower()}-flags' / digest


def _set_prompt_flag(flag: Path, prompt: str) -> None:
  value = SPEAK_BACK_MARKER if SPEAK_BACK_MARKER in prompt else ''
  atomic_write_text(flag, value)


def _claim_prompt_flag(flag: Path, target: str) -> bool:
  if not flag.exists():
    return False
  try:
    value = flag.read_text(encoding='utf-8')
    flag.unlink()
  except OSError as exc:
    raise IntegrationError(f'cannot read {target} prompt state: {exc}') from exc
  return value == SPEAK_BACK_MARKER


def _restore_original(path: Path, record: LifecycleInstallRecord) -> None:
  if record.original_exists:
    atomic_write_text(path, record.original_text)
  else:
    remove_config(path)
