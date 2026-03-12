"""CLI entry point for tts-gateway: `tts serve --provider <name>`."""

from __future__ import annotations

import argparse
import os
import sys

from tts_gateway.config import DEFAULT_MODELS_DIR


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(prog='tts', description='TTS Gateway server')
  sub = parser.add_subparsers(dest='command')

  serve = sub.add_parser('serve', help='Start the TTS gateway server')
  serve.add_argument(
    '--provider', required=True, choices=['kokoro', 'pocket'], help='Primary TTS engine'
  )
  serve.add_argument(
    '--fallback', default='none', help='Fallback engine (default: none)'
  )
  serve.add_argument(
    '--port', type=int, default=8000, help='Server port (default: 8000)'
  )
  serve.add_argument(
    '--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)'
  )
  serve.add_argument(
    '--device',
    default='auto',
    choices=['auto', 'cpu', 'mps', 'cuda'],
    help='Device mode',
  )
  serve.add_argument(
    '--format',
    default='wav',
    choices=['wav', 'mp3'],
    help='Output audio format (default: wav)',
  )
  serve.add_argument('--voice', default=None, help='Default voice name')
  serve.add_argument('--models-dir', default=None, help='Models directory')

  args = parser.parse_args(argv)
  if args.command != 'serve':
    parser.print_help()
    sys.exit(1)

  # Translate CLI flags to env vars before importing app
  os.environ['TTS_PRIMARY_ENGINE'] = args.provider
  os.environ['TTS_FALLBACK_ENGINE'] = args.fallback
  os.environ['TTS_GATEWAY_PORT'] = str(args.port)
  os.environ['TTS_GATEWAY_HOST'] = args.host
  os.environ['TTS_DEVICE_MODE'] = args.device
  os.environ['TTS_OUTPUT_FORMAT'] = args.format

  # Enable the selected provider(s)
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

  import uvicorn

  from tts_gateway.main import create_app

  app = create_app()
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
  main()
