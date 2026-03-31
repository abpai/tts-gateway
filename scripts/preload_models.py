#!/usr/bin/env python3
"""Preload optional model artifacts into the image."""

from __future__ import annotations

import os
import sys


def preload_kokoro(models_dir: str) -> None:
  os.environ.setdefault('HF_HOME', os.path.join(models_dir, 'huggingface'))

  from kokoro import KPipeline

  pipeline = KPipeline(lang_code='a', device='cpu', repo_id='hexgrad/Kokoro-82M')
  del pipeline
  print('  kokoro model preloaded successfully')


def preload_pocket() -> None:
  from pocket_tts import TTSModel

  model = TTSModel.load_model()
  del model
  print('  pocket-tts model preloaded successfully')


def _is_enabled(name: str, default: bool) -> bool:
  raw = os.environ.get(name, str(default)).strip().lower()
  return raw == 'true'


def main() -> None:
  models_dir = os.environ.get(
    'TTS_MODELS_DIR', os.path.expanduser('~/.cache/tts-gateway/models')
  )
  os.makedirs(models_dir, exist_ok=True)

  preload_kokoro_enabled = _is_enabled('PRELOAD_KOKORO', False)
  preload_pocket_enabled = _is_enabled('PRELOAD_POCKET', False)
  if not preload_kokoro_enabled and not preload_pocket_enabled:
    print('No models selected for preloading; skipping.')
    return

  failures: list[str] = []

  if preload_kokoro_enabled:
    print('Preloading Kokoro model...')
    try:
      preload_kokoro(models_dir)
    except Exception as exc:
      failures.append(f'kokoro preload failed: {exc}')

  if preload_pocket_enabled:
    print('Preloading Pocket TTS model...')
    try:
      preload_pocket()
    except Exception as exc:
      failures.append(f'pocket preload failed: {exc}')

  if failures:
    print('Model preloading failed:', file=sys.stderr)
    for failure in failures:
      print(f'  - {failure}', file=sys.stderr)
    raise SystemExit(1)

  print('Model preloading complete.')


if __name__ == '__main__':
  main()
