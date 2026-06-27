"""Shared HTTP helpers for gateway scripts."""

from __future__ import annotations

import http.client
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

NETWORK_ERRORS = (http.client.HTTPException, OSError, TimeoutError)
OpenConnection = Callable[[str], http.client.HTTPConnection]


def request_path_for(base_url: str, path: str) -> str:
  parsed = urlparse(base_url)
  request_path = f'{parsed.path.rstrip("/")}{path}'
  return request_path or '/'


def open_connection(base_url: str) -> http.client.HTTPConnection:
  parsed = urlparse(base_url)
  conn_cls = (
    http.client.HTTPSConnection
    if parsed.scheme == 'https'
    else http.client.HTTPConnection
  )
  return conn_cls(parsed.hostname or '', parsed.port, timeout=180)


def fetch_health(
  base_url: str,
  *,
  open_connection_fn: OpenConnection = open_connection,
) -> dict[str, Any]:
  request_path = request_path_for(base_url, '/health')
  try:
    conn = open_connection_fn(base_url)
  except NETWORK_ERRORS as exc:
    return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
  try:
    conn.request('GET', request_path)
    response = conn.getresponse()
    status = response.status
    body = response.read()
    if not 200 <= status < 300:
      return {'ok': False, 'status': status}
    payload = json.loads(body.decode())
    return health_snapshot(status, payload)
  except json.JSONDecodeError as exc:
    return {'ok': False, 'status': status, 'error': f'JSONDecodeError: {exc}'}
  except NETWORK_ERRORS as exc:
    return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
  finally:
    conn.close()


def health_snapshot(status: int, payload: dict[str, Any]) -> dict[str, Any]:
  snapshot: dict[str, Any] = {
    'ok': True,
    'status': status,
    'primaryEngine': payload.get('primaryEngine'),
    'fallbackEngine': payload.get('fallbackEngine'),
    'engineChain': payload.get('engineChain'),
    'streamFirstChunkMaxChars': payload.get('streamFirstChunkMaxChars'),
    'streamChunkMaxChars': payload.get('streamChunkMaxChars'),
  }
  if 'engines' in payload:
    snapshot['engines'] = payload['engines']
  return snapshot
