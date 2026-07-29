"""Client functions for the `tts speak` command."""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import httpx
from pydantic import BaseModel, ConfigDict, Field


class SpeechCliError(Exception):
  """Report a command-line speech error."""


@dataclass(frozen=True)
class SpeakOptions:
  """Define one speech request."""

  text: str
  base_url: str
  voice: str | None = None
  speed: float | None = None
  output: Path | None = None
  play: bool = True
  stream: bool = True
  ffplay_path: str = 'ffplay'
  connect_timeout: float = 5.0


class HealthReport(BaseModel):
  """Describe the active gateway configuration."""

  model_config = ConfigDict(extra='allow', populate_by_name=True)

  ok: bool
  package_version: str = Field('unknown', alias='packageVersion')
  primary_engine: str = Field(alias='primaryEngine')
  fallback_engine: str | None = Field(None, alias='fallbackEngine')
  output_format: str = Field(alias='outputFormat')
  device_mode: str = Field('unknown', alias='deviceMode')
  default_voice: str | None = Field(None, alias='defaultVoice')
  default_speed: float = Field(1.0, alias='defaultSpeed')
  engine_chain: list[str] = Field(default_factory=list, alias='engineChain')
  engines: dict[str, dict[str, Any]] = Field(default_factory=dict)
  chunk_concurrency: int = Field(alias='chunkConcurrency')
  chunk_max_chars: int = Field(alias='chunkMaxChars')


def read_speak_text(arguments: list[str], stdin: TextIO) -> str:
  """Read text from arguments or standard input."""
  text = ' '.join(arguments).strip()
  if not text and not stdin.isatty():
    text = stdin.read().strip()
  if not text:
    raise SpeechCliError('text is required as an argument or on stdin')
  return text


def speak(options: SpeakOptions, stdout: BinaryIO) -> None:
  """Stream synthesized audio to playback, a file, or standard output."""
  timeout = httpx.Timeout(
    connect=options.connect_timeout,
    read=None,
    write=30,
    pool=options.connect_timeout,
  )
  try:
    with httpx.Client(timeout=timeout) as client:
      with _speech_response(client, options) as response:
        _raise_response_error(response)
        _consume_audio(response, options, stdout)
  except httpx.HTTPError as exc:
    raise SpeechCliError(f'gateway request failed: {exc}') from exc


def fetch_health(base_url: str, timeout: float) -> HealthReport:
  """Fetch and validate gateway health."""
  url = f'{base_url.rstrip("/")}/health'
  try:
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return HealthReport.model_validate(response.json())
  except (httpx.HTTPError, ValueError) as exc:
    raise SpeechCliError(f'health check failed for {url}: {exc}') from exc


def format_config(report: HealthReport, base_url: str) -> str:
  """Format active configuration for terminal output."""
  engine = report.engines.get(report.primary_engine, {})
  model = engine.get('model', report.primary_engine)
  actual_device = engine.get('device')
  device = (
    actual_device if actual_device not in {None, 'unknown'} else report.device_mode
  )
  values = [
    ('Gateway', base_url.rstrip('/')),
    ('Version', report.package_version),
    ('Model / engine', model),
    ('Fallback engine', report.fallback_engine or 'none'),
    ('Engine chain', ', '.join(report.engine_chain)),
    ('Device', device),
    ('Default voice', report.default_voice or 'engine default'),
    ('Default speed', report.default_speed),
    ('Output format', report.output_format),
    ('Streaming', 'enabled'),
    ('Playback', 'enabled'),
    ('Chunk concurrency', report.chunk_concurrency),
    ('Chunk size', report.chunk_max_chars),
  ]
  width = max(len(label) for label, _ in values)
  return '\n'.join(f'{label:<{width}}  {value}' for label, value in values)


def _speech_response(
  client: httpx.Client,
  options: SpeakOptions,
) -> AbstractContextManager[httpx.Response]:
  payload: dict[str, str | float] = {'text': options.text}
  if options.voice is not None:
    payload['voice'] = options.voice
  if options.speed is not None:
    payload['speed'] = options.speed
  endpoint = '/tts/stream' if options.stream else '/v1/speech'
  url = f'{options.base_url.rstrip("/")}{endpoint}'
  if options.stream:
    return client.stream('POST', url, json=payload)
  form_payload = {name: str(value) for name, value in payload.items()}
  return client.stream('POST', url, data=form_payload)


def _raise_response_error(response: httpx.Response) -> None:
  if not response.is_error:
    return
  detail = response.read().decode(errors='replace').strip()
  message = detail or response.reason_phrase
  raise SpeechCliError(f'gateway returned HTTP {response.status_code}: {message}')


def _consume_audio(
  response: httpx.Response,
  options: SpeakOptions,
  stdout: BinaryIO,
) -> None:
  with ExitStack() as stack:
    output = _open_output(stack, options, stdout)
    player = _open_player(options) if options.play else None
    try:
      for chunk in response.iter_bytes():
        output.write(chunk)
        if player is not None and player.stdin is not None:
          player.stdin.write(chunk)
    finally:
      _finish_player(player)


def _open_output(
  stack: ExitStack,
  options: SpeakOptions,
  stdout: BinaryIO,
) -> BinaryIO:
  if options.output is not None:
    return stack.enter_context(options.output.open('wb'))
  if options.play:
    return stack.enter_context(Path(os.devnull).open('wb'))
  return stdout


def _open_player(options: SpeakOptions) -> subprocess.Popen[bytes]:
  ffplay_path = shutil.which(options.ffplay_path)
  if ffplay_path is None:
    raise SpeechCliError(f'ffplay executable not found: {options.ffplay_path}')
  return subprocess.Popen(
    [
      ffplay_path,
      '-autoexit',
      '-nodisp',
      '-loglevel',
      'error',
      '-i',
      'pipe:0',
    ],
    stdin=subprocess.PIPE,
  )


def _finish_player(player: subprocess.Popen[bytes] | None) -> None:
  if player is None:
    return
  if player.stdin is not None:
    player.stdin.close()
  return_code = player.wait()
  if return_code != 0:
    raise SpeechCliError(f'ffplay exited with status {return_code}')
