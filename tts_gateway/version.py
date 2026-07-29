"""Installed package version lookup."""

from __future__ import annotations

import importlib.metadata


def package_version() -> str:
  try:
    return importlib.metadata.version('tts-gateway')
  except importlib.metadata.PackageNotFoundError:
    return 'unknown'
