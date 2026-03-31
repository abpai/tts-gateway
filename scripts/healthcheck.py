#!/usr/bin/env python3
"""Docker healthcheck for the running gateway."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def main() -> int:
  port = os.getenv('TTS_GATEWAY_PORT', '8080').strip() or '8080'
  url = f'http://127.0.0.1:{port}/health'
  try:
    with urlopen(url, timeout=3) as response:
      if response.status != 200:
        return 1
      payload = json.load(response)
  except (HTTPError, URLError, OSError, ValueError):
    return 1

  return 0 if payload.get('ok') is True else 1


if __name__ == '__main__':
  raise SystemExit(main())
