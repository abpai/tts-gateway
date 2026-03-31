"""CLI entry point for tts-gateway: `tts serve` and `tts worker`."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from tts_gateway.config import DEFAULT_MODELS_DIR

_NOISY_LOGGERS = ('httpcore', 'httpx', 'urllib3', 'filelock')
_QUIET_LOGGERS = ('huggingface_hub',)  # suppress HF token nag


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
    '--provider', required=True, choices=['kokoro', 'pocket'], help='Primary TTS engine'
  )
  parser.add_argument(
    '--fallback', default='none', help='Fallback engine (default: none)'
  )
  parser.add_argument(
    '--device',
    default='auto',
    choices=['auto', 'cpu', 'mps', 'cuda'],
    help='Device mode',
  )
  parser.add_argument(
    '--format',
    default='wav',
    choices=['wav', 'mp3'],
    help='Output audio format (default: wav)',
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

  if args.models_dir:
    os.environ['TTS_MODELS_DIR'] = args.models_dir
  elif 'TTS_MODELS_DIR' not in os.environ:
    os.environ['TTS_MODELS_DIR'] = os.path.expanduser(DEFAULT_MODELS_DIR)

  if args.voice:
    os.environ['TTS_DEFAULT_VOICE'] = args.voice

  if args.chunk_size:
    os.environ['TTS_CHUNK_MAX_CHARS'] = str(args.chunk_size)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(prog='tts', description='TTS Gateway server')
  sub = parser.add_subparsers(dest='command')

  # --- serve ---
  serve = sub.add_parser('serve', help='Start the TTS gateway server')
  _add_common_args(serve)
  serve.add_argument(
    '--port', type=int, default=8000, help='Server port (default: 8000)'
  )
  serve.add_argument(
    '--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)'
  )

  # --- worker ---
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

  args = parser.parse_args(argv)

  if args.command == 'serve':
    _run_serve(args)
  elif args.command == 'worker':
    _run_worker(args)
  else:
    parser.print_help()
    sys.exit(1)


def _run_serve(args: argparse.Namespace) -> None:
  _set_common_env(args)
  os.environ['TTS_GATEWAY_PORT'] = str(args.port)
  os.environ['TTS_GATEWAY_HOST'] = args.host
  _configure_logging(args.debug)

  import uvicorn

  from tts_gateway.main import create_app

  app = create_app()
  uvicorn.run(
    app,
    host=args.host,
    port=args.port,
    log_level='debug' if args.debug else 'info',
  )


def _run_worker(args: argparse.Namespace) -> None:
  import asyncio
  from pathlib import Path

  _set_common_env(args)
  _configure_logging(args.debug)

  from tts_gateway.config import load_config
  from tts_gateway.gateway import TtsGateway
  from tts_gateway.jobs.store import JobStore
  from tts_gateway.jobs.worker import run_worker

  config = load_config()
  gateway = TtsGateway(config)
  data_dir = Path(config.data_dir)
  artifacts_dir = data_dir / 'artifacts'
  artifacts_dir.mkdir(parents=True, exist_ok=True)
  store = JobStore(data_dir / 'jobs.db')

  try:
    asyncio.run(
      run_worker(
        store,
        gateway.engine_list(),
        artifacts_dir,
        poll_seconds=args.poll_interval,
        concurrency=gateway.chunk_concurrency(),
        engine_timeout=config.engine_timeout_seconds,
        ffmpeg_path=config.ffmpeg_path,
        once=args.once,
      )
    )
  finally:
    store.close()


if __name__ == '__main__':
  main()
