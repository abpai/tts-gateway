"""Client functions for the `tts speak` command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import httpx
from pydantic import BaseModel, ConfigDict, Field

_FORMAT_CONTENT_TYPES = {
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
}

# ffplay demuxer names by response content type; a known format is passed via
# -f so ffplay skips stream probing. An unknown content type falls back to
# probing, and a missing header assumes mp3 (the streaming route's format).
_CONTENT_TYPE_FFPLAY_FORMATS = {
  'audio/mpeg': 'mp3',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
}

# Fallback PCM parameters if the gateway ever omits an X-TTS-* PCM header;
# match the server's own kokoro/base-engine defaults (24kHz mono 16-bit).
_DEFAULT_PCM_FORMAT = 's16le'
_DEFAULT_SAMPLE_RATE = '24000'
_DEFAULT_CHANNELS = '1'


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
  play_only: bool = False
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


def read_speak_text(arguments: list[str], stdin: TextIO | None) -> str:
  """Read text from arguments or standard input."""
  text = ' '.join(arguments).strip()
  if not text and stdin is not None and not stdin.isatty():
    text = stdin.read().strip()
  if not text:
    raise SpeechCliError('text is required as an argument or on stdin')
  return text


class _Player:
  """Own the ffplay subprocess used for streamed playback."""

  def __init__(self, process: subprocess.Popen[bytes]) -> None:
    self._process = process

  @classmethod
  def start(cls, ffplay_path: str, format_args: tuple[str, ...]) -> _Player:
    process = subprocess.Popen(
      [
        ffplay_path,
        '-autoexit',
        '-nodisp',
        '-loglevel',
        'error',
        *format_args,
        '-i',
        'pipe:0',
      ],
      stdin=subprocess.PIPE,
    )
    return cls(process)

  def write(self, chunk: bytes) -> bool:
    """Write one chunk to ffplay's stdin; return False on a broken pipe (early stop)."""
    assert self._process.stdin is not None
    try:
      self._process.stdin.write(chunk)
      self._process.stdin.flush()
    except BrokenPipeError:
      return False
    return True

  def finish(self, *, body_completed: bool) -> None:
    """Wait for the player and surface a real failure, without masking earlier errors.

    A non-zero exit only counts as an error if the streaming body finished
    without incident; a negative return code (killed by signal, e.g. Ctrl-C)
    or an early user quit is a normal stop, not an error.

    No timeout on wait(): ffplay legitimately keeps playing buffered audio
    after EOF, and a timeout would kill valid playback.
    """
    if self._process.stdin is not None:
      try:
        self._process.stdin.close()
      except BrokenPipeError:
        pass
    return_code = self._process.wait()
    if body_completed and return_code > 0:
      raise SpeechCliError(f'ffplay exited with status {return_code}')


def speak(options: SpeakOptions, stdout: BinaryIO) -> None:
  """Stream synthesized audio to playback, a file, or standard output.

  Resolves the output sink and the ffplay binary before sending the request,
  so a bad --output path or a missing ffplay binary fails before synthesis
  runs. The ffplay process itself is spawned only after the response headers
  are validated, so a gateway error never leaves an orphaned player process,
  and PCM playback can size its decoder from the response's PCM headers.
  """
  timeout = httpx.Timeout(
    connect=options.connect_timeout,
    read=300,
    write=30,
    pool=options.connect_timeout,
  )
  with ExitStack() as stack:
    output = _open_output(stack, options, stdout)
    flush_output = output is stdout
    ffplay_path = _resolve_ffplay(options.ffplay_path) if options.play else None
    play_pcm = ffplay_path is not None and _wants_pcm(options, stdout)
    request_format = 'pcm' if play_pcm else None
    player: _Player | None = None
    body_completed = False
    try:
      with httpx.Client(timeout=timeout) as client:
        with _speech_response(
          client, options, request_format=request_format
        ) as response:
          _raise_response_error(response)
          _warn_format_mismatch(response, options)
          if ffplay_path is not None:
            format_args = (
              _pcm_ffplay_args(response) if play_pcm else _encoded_ffplay_args(response)
            )
            player = _Player.start(ffplay_path, format_args)
          body_completed = _consume_audio(
            response, output, player, flush_output=flush_output
          )
    except httpx.HTTPError as exc:
      raise SpeechCliError(f'gateway request failed: {exc}') from exc
    finally:
      if player is not None:
        player.finish(body_completed=body_completed)


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
    ('Chunk concurrency', report.chunk_concurrency),
    ('Chunk size', report.chunk_max_chars),
  ]
  width = max(len(label) for label, _ in values)
  return '\n'.join(f'{label:<{width}}  {value}' for label, value in values)


def _speech_response(
  client: httpx.Client,
  options: SpeakOptions,
  *,
  request_format: str | None = None,
) -> AbstractContextManager[httpx.Response]:
  payload: dict[str, str | float] = {'text': options.text}
  if options.voice is not None:
    payload['voice'] = options.voice
  if options.speed is not None:
    payload['speed'] = options.speed
  if request_format is not None:
    payload['format'] = request_format
  endpoint = '/v1/speech/stream' if options.stream else '/v1/speech'
  url = f'{options.base_url.rstrip("/")}{endpoint}'
  return client.stream('POST', url, json=payload)


def _raise_response_error(response: httpx.Response) -> None:
  if not response.is_error:
    return
  detail = response.read().decode(errors='replace').strip()
  message = detail or response.reason_phrase
  raise SpeechCliError(f'gateway returned HTTP {response.status_code}: {message}')


def _warn_format_mismatch(response: httpx.Response, options: SpeakOptions) -> None:
  """Warn on stderr when --output's suffix disagrees with the response format.

  Only applies to non-streaming requests: /v1/speech returns the gateway's
  configured output format, which may not match the suffix the caller chose.
  """
  if options.stream or options.output is None:
    return
  expected = _FORMAT_CONTENT_TYPES.get(options.output.suffix.lower())
  if expected is None:
    return
  content_type = response.headers.get('content-type', '').split(';')[0].strip().lower()
  if content_type and content_type != expected:
    print(
      f'warning: gateway returned {content_type}, but output path is '
      f'{options.output.suffix}',
      file=sys.stderr,
    )


def _wants_pcm(options: SpeakOptions, stdout: BinaryIO) -> bool:
  """True for the play-only path: play enabled, no --output, stdout is a tty.

  `play_only` skips the tty sniff: callers that only ever play (the agent
  integrations run detached with stdout on /dev/null) still get raw PCM,
  which starts fast and never loses leading audio to stream probing.

  Streaming-only: /v1/speech (non-streaming) has no format field, so
  --no-stream always requests mp3.
  """
  if not options.stream or not options.play or options.output is not None:
    return False
  if options.play_only:
    return True
  return bool(getattr(stdout, 'isatty', lambda: False)())


def _encoded_ffplay_args(response: httpx.Response) -> tuple[str, ...]:
  """Build ffplay args for an encoded (mp3/wav) stream from its Content-Type.

  The non-streaming /v1/speech route returns the gateway's configured format,
  which may be wav — telling ffplay the wrong format would break playback.

  No -fflags nobuffer here: on encoded live streams it makes ffplay discard
  the data consumed during stream analysis, audibly cutting off the first
  words. Raw PCM playback keeps it, where analysis consumes nothing.
  """
  content_type = response.headers.get('content-type', '').split(';')[0].strip().lower()
  if not content_type:
    return ('-f', 'mp3')
  ffplay_format = _CONTENT_TYPE_FFPLAY_FORMATS.get(content_type)
  if ffplay_format is None:
    return ()  # unknown format: let ffplay probe
  return ('-f', ffplay_format)


def _pcm_ffplay_args(response: httpx.Response) -> tuple[str, ...]:
  """Build ffplay's raw-PCM decode args from the response's X-TTS-* PCM headers.

  ffplay has no -ac option (that is the ffmpeg CLI); the channel count must be
  expressed as a -ch_layout name.
  """
  pcm_format = response.headers.get('x-tts-pcm-format', _DEFAULT_PCM_FORMAT)
  sample_rate = response.headers.get('x-tts-sample-rate', _DEFAULT_SAMPLE_RATE)
  channels = response.headers.get('x-tts-channels', _DEFAULT_CHANNELS)
  layout = {'1': 'mono', '2': 'stereo'}.get(channels, f'{channels}c')
  return (
    '-f',
    pcm_format,
    '-ar',
    sample_rate,
    '-ch_layout',
    layout,
    '-fflags',
    'nobuffer',
  )


def _consume_audio(
  response: httpx.Response,
  output: BinaryIO,
  player: _Player | None,
  *,
  flush_output: bool,
) -> bool:
  """Write streamed audio chunks to the output sink and player.

  Returns True if the response body was read to completion. Stops cleanly
  (returning False) if a downstream pipe closes early, e.g. `head` truncating
  piped stdout, or the user quitting ffplay.
  """
  for chunk in response.iter_bytes():
    try:
      output.write(chunk)
      if flush_output:
        output.flush()
    except BrokenPipeError:
      return False
    if player is not None and not player.write(chunk):
      return False
  return True


def _open_output(
  stack: ExitStack,
  options: SpeakOptions,
  stdout: BinaryIO,
) -> BinaryIO:
  if options.output is not None:
    try:
      return stack.enter_context(options.output.open('wb'))
    except OSError as exc:
      raise SpeechCliError(f'cannot open output file: {exc}') from exc
  is_tty = getattr(stdout, 'isatty', lambda: False)()
  if options.play:
    if is_tty or options.play_only:
      return stack.enter_context(Path(os.devnull).open('wb'))
    return stdout  # tee: redirected stdout also receives the audio bytes
  if is_tty:
    raise SpeechCliError(
      'refusing to write binary audio to a terminal; use --output or redirect stdout'
    )
  return stdout


def _resolve_ffplay(ffplay_path: str) -> str:
  """Resolve the ffplay executable, failing fast before the network request."""
  resolved = shutil.which(ffplay_path)
  if resolved is None:
    raise SpeechCliError(f'ffplay executable not found: {ffplay_path}')
  return resolved
