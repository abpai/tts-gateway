"""CLI entry point for tts-gateway."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from tts_gateway.config import DEFAULT_MODELS_DIR
from tts_gateway.integrations import claude as claude_integration
from tts_gateway.integrations import codex as codex_integration
from tts_gateway.integrations import instructions
from tts_gateway.integrations.common import (
  claude_memory_path,
  claude_settings_path,
  codex_agents_path,
  codex_hooks_path,
  read_event_payload,
  resolve_tts_executable,
  speak_file,
  state_root,
)
from tts_gateway.speech_cli import (
  SpeakOptions,
  SpeechCliError,
  fetch_health,
  format_config,
  read_speak_text,
  speak,
)
from tts_gateway.version import package_version

_NOISY_LOGGERS = ('httpcore', 'httpx', 'urllib3', 'filelock')
_QUIET_LOGGERS = ('huggingface_hub',)  # suppress HF token nag


def _suppress_third_party_warnings() -> None:
  """Mute noisy torch/kokoro/HF warnings that fire on every import or inference."""
  warnings.filterwarnings('ignore', category=UserWarning, module=r'torch\.')
  warnings.filterwarnings('ignore', category=UserWarning, module=r'kokoro\.')
  warnings.filterwarnings('ignore', category=FutureWarning, module=r'torch\.')
  warnings.filterwarnings('ignore', message='.*unauthenticated requests.*HF.*')


def _configure_logging(debug: bool) -> None:
  logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
  )
  logging.getLogger('tts_gateway').setLevel(logging.DEBUG if debug else logging.INFO)
  if not debug:
    for name in _NOISY_LOGGERS:
      logging.getLogger(name).setLevel(logging.WARNING)
    for name in _QUIET_LOGGERS:
      logging.getLogger(name).setLevel(logging.ERROR)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
  """Add flags shared between serve and worker subcommands."""
  parser.add_argument(
    '--provider',
    required=True,
    choices=['kokoro', 'pocket', 'cosyvoice'],
    help='Primary TTS engine',
  )
  parser.add_argument(
    '--fallback',
    default='none',
    choices=['none', 'kokoro', 'pocket', 'cosyvoice'],
    help='Fallback engine (default: none)',
  )
  parser.add_argument(
    '--device',
    default='auto',
    choices=['auto', 'cpu', 'mps', 'cuda'],
    help='Device mode',
  )
  parser.add_argument(
    '--format',
    default='mp3',
    choices=['wav', 'mp3'],
    help='Output audio format (default: mp3)',
  )
  parser.add_argument('--voice', default=None, help='Default voice name')
  parser.add_argument('--models-dir', default=None, help='Models directory')
  parser.add_argument(
    '--chunk-size',
    type=int,
    default=None,
    help='Max characters per text chunk (default: 500)',
  )
  parser.add_argument(
    '--debug',
    action='store_true',
    help='Enable verbose logging with chunk progress bar.',
  )


def _set_common_env(args: argparse.Namespace) -> None:
  """Translate shared CLI flags to env vars before importing app modules."""
  os.environ['TTS_PRIMARY_ENGINE'] = args.provider
  os.environ['TTS_FALLBACK_ENGINE'] = args.fallback
  os.environ['TTS_DEVICE_MODE'] = args.device
  os.environ['TTS_OUTPUT_FORMAT'] = args.format

  os.environ['KOKORO_TTS_ENABLED'] = str(
    args.provider == 'kokoro' or args.fallback == 'kokoro'
  ).lower()
  os.environ['POCKET_TTS_ENABLED'] = str(
    args.provider == 'pocket' or args.fallback == 'pocket'
  ).lower()
  os.environ['COSYVOICE_TTS_ENABLED'] = str(
    args.provider == 'cosyvoice' or args.fallback == 'cosyvoice'
  ).lower()

  if args.models_dir:
    os.environ['TTS_MODELS_DIR'] = args.models_dir
  elif 'TTS_MODELS_DIR' not in os.environ:
    os.environ['TTS_MODELS_DIR'] = os.path.expanduser(DEFAULT_MODELS_DIR)

  if args.voice:
    os.environ['TTS_DEFAULT_VOICE'] = args.voice

  if args.chunk_size:
    os.environ['TTS_CHUNK_MAX_CHARS'] = str(args.chunk_size)


def build_parser() -> argparse.ArgumentParser:
  """Build the command-line parser."""
  parser = argparse.ArgumentParser(prog='tts', description='TTS Gateway server')
  parser.add_argument(
    '--version',
    action='version',
    version=f'%(prog)s {package_version()}',
  )
  sub = parser.add_subparsers(dest='command')
  _add_serve_parser(sub)
  _add_worker_parser(sub)
  _add_speak_parser(sub)
  _add_status_parsers(sub)
  _add_integrate_parser(sub)
  sub.add_parser('update', help='Update the installed tts-gateway tool')
  return parser


def _add_serve_parser(sub: argparse._SubParsersAction) -> None:
  serve = sub.add_parser('serve', help='Start the TTS gateway server')
  _add_common_args(serve)
  serve.add_argument(
    '--port', type=int, default=45123, help='Server port (default: 45123)'
  )
  serve.add_argument(
    '--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)'
  )


def _add_worker_parser(sub: argparse._SubParsersAction) -> None:
  worker = sub.add_parser('worker', help='Run the background job worker')
  _add_common_args(worker)
  worker.add_argument(
    '--once',
    action='store_true',
    help='Process one job and exit',
  )
  worker.add_argument(
    '--poll-interval',
    type=float,
    default=1.0,
    help='Seconds between queue polls (default: 1.0)',
  )


def _add_speak_parser(sub: argparse._SubParsersAction) -> None:
  speak_parser = sub.add_parser('speak', help='Speak text through the gateway')
  speak_parser.add_argument(
    'text', nargs='*', help='Text to speak; reads stdin if omitted'
  )
  _add_gateway_args(speak_parser, timeout_label='Connection')
  speak_parser.add_argument('--voice', help='Voice name')
  speak_parser.add_argument('--speed', type=float, help='Speech speed')
  speak_parser.add_argument('-o', '--output', type=Path, help='Write audio to a file')
  _add_speak_behavior_args(speak_parser)


def _add_speak_behavior_args(parser: argparse.ArgumentParser) -> None:
  """Add playback and streaming options."""
  parser.add_argument(
    '--play',
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      'Play audio while it arrives (default: enabled); without --output, '
      'redirected stdout also receives the audio bytes'
    ),
  )
  parser.add_argument(
    '--stream',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Use streaming synthesis (default: enabled)',
  )
  parser.add_argument(
    '--ffplay-path',
    default='ffplay',
    help='ffplay executable (default: ffplay)',
  )


def _add_status_parsers(sub: argparse._SubParsersAction) -> None:
  health = sub.add_parser('health', help='Check gateway health')
  _add_gateway_args(health)
  config = sub.add_parser('config', help='Show the active gateway configuration')
  _add_gateway_args(config)


def _add_integrate_parser(sub: argparse._SubParsersAction) -> None:
  """Add coding-agent integration commands."""
  integrate = sub.add_parser(
    'integrate',
    help='Install Codex and Claude Code speech integrations',
  )
  actions = integrate.add_subparsers(dest='integrate_action')
  for action in ('install', 'uninstall'):
    parser = actions.add_parser(action, help=f'{action.title()} an integration')
    parser.add_argument('target', choices=['codex', 'claude', 'all'])
  status = actions.add_parser('status', help='Show integration status')
  status.add_argument('target', nargs='?', choices=['codex', 'claude', 'all'])
  integrate.add_argument(
    '--codex-event',
    action='store_true',
    help=argparse.SUPPRESS,
  )
  integrate.add_argument(
    '--claude-event',
    action='store_true',
    help=argparse.SUPPRESS,
  )
  integrate.add_argument('--speak-file', type=Path, help=argparse.SUPPRESS)


def _add_gateway_args(
  parser: argparse.ArgumentParser, *, timeout_label: str = 'Request'
) -> None:
  """Add gateway connection options."""
  parser.add_argument(
    '--base-url',
    default=os.getenv('TTS_GATEWAY_URL', 'http://127.0.0.1:45123'),
    help='Gateway URL (default: $TTS_GATEWAY_URL or http://127.0.0.1:45123)',
  )
  parser.add_argument(
    '--timeout',
    type=float,
    default=5.0,
    help=f'{timeout_label} timeout in seconds (default: 5)',
  )


def main(argv: list[str] | None = None) -> None:
  parser = build_parser()
  args = parser.parse_args(argv)

  try:
    if args.command == 'serve':
      _run_serve(args)
    elif args.command == 'worker':
      _run_worker(args)
    elif args.command == 'speak':
      _run_speak(args)
    elif args.command == 'health':
      _run_health(args)
    elif args.command == 'config':
      _run_config(args)
    elif args.command == 'integrate':
      _run_integrate(args)
    elif args.command == 'update':
      _run_update()
    else:
      parser.print_help()
      parser.exit(1)
  except SpeechCliError as exc:
    parser.exit(1, f'error: {exc}\n')
  except KeyboardInterrupt:
    parser.exit(130)


def _run_speak(args: argparse.Namespace) -> None:
  text = read_speak_text(args.text, sys.stdin)
  options = SpeakOptions(
    text=text,
    base_url=args.base_url,
    voice=args.voice,
    speed=args.speed,
    output=args.output,
    play=args.play,
    stream=args.stream,
    ffplay_path=args.ffplay_path,
    connect_timeout=args.timeout,
  )
  speak(options, sys.stdout.buffer)


def _run_health(args: argparse.Namespace) -> None:
  report = fetch_health(args.base_url, args.timeout)
  if not report.ok:
    raise SpeechCliError('gateway reported an unhealthy status')
  print(f'healthy  {args.base_url.rstrip("/")}')


def _run_config(args: argparse.Namespace) -> None:
  report = fetch_health(args.base_url, args.timeout)
  print(format_config(report, args.base_url))


def _run_integrate(args: argparse.Namespace) -> None:
  """Run one coding-agent integration command."""
  if args.codex_event:
    _run_integration_event('codex')
  elif args.claude_event:
    _run_integration_event('claude')
  elif args.speak_file is not None:
    with Path(os.devnull).open('wb') as output:
      speak_file(args.speak_file, output)
  elif args.integrate_action == 'install':
    _change_integrations(args.target, install=True)
  elif args.integrate_action == 'uninstall':
    _change_integrations(args.target, install=False)
  elif args.integrate_action == 'status':
    _show_integration_status(args.target or 'all')
  else:
    raise SpeechCliError('integration action is required')


def _change_integrations(target: str, *, install: bool) -> None:
  root = state_root()
  executable = resolve_tts_executable()
  results: list[tuple[str, bool]] = []
  try:
    for name in _integration_targets(target):
      changed = _change_integration(name, install, executable, root)
      results.append((name, changed))
  except SpeechCliError:
    if install:
      _roll_back_installs(results, executable, root)
    raise
  action = 'installed' if install else 'uninstalled'
  for name, changed in results:
    print(f'{name:<6}  {action if changed else "unchanged"}')
  if install and ('codex', True) in results:
    print('Run /hooks in Codex to review and trust the new hooks.')


def _roll_back_installs(
  results: list[tuple[str, bool]],
  executable: str,
  root: Path,
) -> None:
  for name, changed in reversed(results):
    if changed:
      _change_integration(name, False, executable, root)


def _change_integration(
  target: str,
  install: bool,
  executable: str,
  root: Path,
) -> bool:
  if target == 'codex':
    paths = (codex_hooks_path(), root / 'codex.json')
    if install:
      hooks_changed = codex_integration.install(*paths, executable, root)
      return instructions.install(codex_agents_path()) or hooks_changed
    hooks_changed = codex_integration.uninstall(*paths)
    return instructions.uninstall(codex_agents_path()) or hooks_changed
  paths = (claude_settings_path(), root / 'claude.json')
  if install:
    hooks_changed = claude_integration.install(*paths, executable, root)
    return instructions.install(claude_memory_path()) or hooks_changed
  hooks_changed = claude_integration.uninstall(*paths)
  return instructions.uninstall(claude_memory_path()) or hooks_changed


def _show_integration_status(target: str) -> None:
  root = state_root()
  for name in _integration_targets(target):
    if name == 'codex':
      result = codex_integration.status(codex_hooks_path(), root / 'codex.json')
    else:
      result = claude_integration.status(
        claude_settings_path(),
        root / 'claude.json',
      )
    print(f'{name:<6}  {result}')


def _run_integration_event(target: str) -> None:
  root = state_root()
  payload = read_event_payload(sys.stdin)
  if target == 'codex':
    codex_integration.handle_event(payload, root / 'codex.json', root)
    print('{}')
    return
  claude_integration.handle_event(payload, root / 'claude.json', root)


def _integration_targets(target: str) -> tuple[str, ...]:
  return ('codex', 'claude') if target == 'all' else (target,)


def _run_update() -> None:
  uv_path = shutil.which('uv')
  if uv_path is None:
    raise SpeechCliError('uv is required to update tts-gateway')
  result = subprocess.run(
    [
      uv_path,
      'tool',
      'upgrade',
      'tts-gateway',
      '--reinstall-package',
      'tts-gateway',
    ],
    check=False,
  )
  if result.returncode != 0:
    raise SpeechCliError('uv could not update tts-gateway')
  tts_path = shutil.which('tts')
  if tts_path is not None:
    # Report the freshly installed binary's version, not this process's
    # (potentially now-stale) in-memory package_version().
    subprocess.run([tts_path, '--version'], check=False)


def _run_serve(args: argparse.Namespace) -> None:
  _set_common_env(args)
  os.environ['TTS_GATEWAY_PORT'] = str(args.port)
  os.environ['TTS_GATEWAY_HOST'] = args.host
  _suppress_third_party_warnings()
  _configure_logging(args.debug)

  import uvicorn

  from tts_gateway.routes import create_app

  app = create_app()
  uvicorn.run(
    app,
    host=args.host,
    port=args.port,
    log_level='debug' if args.debug else 'info',
  )


def _run_worker(args: argparse.Namespace) -> None:
  import asyncio

  _set_common_env(args)
  _suppress_third_party_warnings()
  _configure_logging(args.debug)

  from tts_gateway.config import load_config
  from tts_gateway.runtime import JobRuntime, run_worker_loop

  config = load_config()
  runtime = JobRuntime(config)

  try:
    asyncio.run(
      run_worker_loop(
        runtime,
        poll_seconds=args.poll_interval,
        once=args.once,
      )
    )
  finally:
    runtime.close()


if __name__ == '__main__':
  main()
