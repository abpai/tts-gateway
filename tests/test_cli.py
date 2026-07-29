from __future__ import annotations

import json
import subprocess
import urllib.parse
from contextlib import ExitStack
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, BinaryIO, cast

import httpx
import pytest

from tts_gateway import cli
from tts_gateway import speech_cli as speech_cli_module
from tts_gateway.speech_cli import (
  HealthReport,
  SpeakOptions,
  SpeechCliError,
  _consume_audio,
  _finish_player,
  _open_output,
  _open_player,
  _raise_response_error,
  _speech_response,
  _warn_format_mismatch,
  fetch_health,
  format_config,
  read_speak_text,
  speak,
)


class _FakeStdout(BytesIO):
  """A BytesIO that can report an arbitrary isatty() value, like a real stream."""

  def __init__(self, tty: bool) -> None:
    super().__init__()
    self._tty = tty

  def isatty(self) -> bool:
    return self._tty


class _FakePlayer:
  """A minimal stand-in for subprocess.Popen[bytes] for _finish_player tests."""

  def __init__(self, return_code: int, stdin: Any | None = None) -> None:
    self.returncode = return_code
    self.stdin = stdin
    self._return_code = return_code

  def wait(self) -> int:
    return self._return_code


class _BrokenStdin:
  """A file-like object whose writes always raise BrokenPipeError."""

  def write(self, data: bytes) -> int:
    raise BrokenPipeError

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass


def _health_report(**overrides: object) -> HealthReport:
  values: dict[str, object] = {
    'ok': True,
    'packageVersion': '1.2.0',
    'primaryEngine': 'kokoro',
    'fallbackEngine': None,
    'outputFormat': 'mp3',
    'deviceMode': 'auto',
    'defaultVoice': 'af_heart',
    'defaultSpeed': 1.0,
    'engineChain': ['kokoro'],
    'engines': {
      'kokoro': {
        'model': 'hexgrad/Kokoro-82M',
        'device': 'mps',
      }
    },
    'chunkConcurrency': 4,
    'chunkMaxChars': 500,
  }
  values.update(overrides)
  return HealthReport.model_validate(values)


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
  with pytest.raises(SystemExit) as excinfo:
    cli.main(['--version'])

  assert excinfo.value.code == 0
  assert capsys.readouterr().out == f'tts {cli.package_version()}\n'


def test_speak_defaults_to_streaming_playback() -> None:
  args = cli.build_parser().parse_args(['speak', 'hi there'])

  assert args.base_url == 'http://127.0.0.1:45123'
  assert args.play is True
  assert args.stream is True
  assert args.output is None


def test_speak_supports_no_play_and_no_stream() -> None:
  args = cli.build_parser().parse_args(
    ['speak', '--no-play', '--no-stream', 'hi there']
  )

  assert args.play is False
  assert args.stream is False


def test_read_speak_text_from_stdin() -> None:
  assert read_speak_text([], StringIO('hi from stdin\n')) == 'hi from stdin'


def test_read_speak_text_rejects_empty_input() -> None:
  with pytest.raises(SpeechCliError, match='text is required'):
    read_speak_text([], StringIO('  \n'))


def test_read_speak_text_prefers_arguments_over_stdin() -> None:
  assert read_speak_text(['hi', 'there'], StringIO('ignored\n')) == 'hi there'


def test_read_speak_text_raises_when_stdin_is_none_and_no_args() -> None:
  with pytest.raises(SpeechCliError, match='text is required'):
    read_speak_text([], None)


def test_stream_request_sends_json() -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == '/tts/stream'
    assert json.loads(request.content) == {
      'text': 'hello',
      'voice': 'af_heart',
      'speed': 1.25,
    }
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  options = SpeakOptions('hello', 'http://gateway', voice='af_heart', speed=1.25)
  with httpx.Client(transport=transport) as client:
    with _speech_response(client, options) as response:
      assert response.read() == b'audio'


def test_form_request_sends_urlencoded_body() -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == '/v1/speech'
    assert request.headers['content-type'] == 'application/x-www-form-urlencoded'
    body = urllib.parse.parse_qs(request.content.decode())
    assert body == {'text': ['hello'], 'voice': ['af_heart'], 'speed': ['1.25']}
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  options = SpeakOptions(
    'hello', 'http://gateway', voice='af_heart', speed=1.25, stream=False
  )
  with httpx.Client(transport=transport) as client:
    with _speech_response(client, options) as response:
      assert response.read() == b'audio'


def test_raise_response_error_surfaces_detail() -> None:
  response = httpx.Response(500, content=b'engine crashed')

  with pytest.raises(SpeechCliError, match='engine crashed'):
    _raise_response_error(response)


def test_warn_format_mismatch_prints_warning_for_no_stream(
  capsys: pytest.CaptureFixture[str],
) -> None:
  response = httpx.Response(200, headers={'content-type': 'audio/wav'})
  options = SpeakOptions(
    'hello', 'http://gateway', output=Path('speech.mp3'), stream=False
  )

  _warn_format_mismatch(response, options)

  assert 'audio/wav' in capsys.readouterr().err


def test_warn_format_mismatch_silent_when_streaming(
  capsys: pytest.CaptureFixture[str],
) -> None:
  response = httpx.Response(200, headers={'content-type': 'audio/wav'})
  options = SpeakOptions(
    'hello', 'http://gateway', output=Path('speech.mp3'), stream=True
  )

  _warn_format_mismatch(response, options)

  assert capsys.readouterr().err == ''


def test_no_play_writes_audio_to_stdout() -> None:
  response = httpx.Response(200, content=b'audio')
  stdout = BytesIO()

  completed = _consume_audio(response, stdout, None)

  assert completed is True
  assert stdout.getvalue() == b'audio'


def test_output_file_receives_audio(tmp_path: Path) -> None:
  output_path = tmp_path / 'speech.mp3'
  response = httpx.Response(200, content=b'audio')

  with output_path.open('wb') as output:
    completed = _consume_audio(response, output, None)

  assert completed is True
  assert output_path.read_bytes() == b'audio'


def test_consume_audio_stops_cleanly_on_broken_player_pipe() -> None:
  response = httpx.Response(200, content=b'audio')
  player = _FakePlayer(0, stdin=_BrokenStdin())

  completed = _consume_audio(
    response, BytesIO(), cast('subprocess.Popen[bytes]', player)
  )

  assert completed is False


def test_consume_audio_stops_cleanly_on_broken_output_pipe() -> None:
  class _BrokenOutput:
    def write(self, data: bytes) -> int:
      raise BrokenPipeError

  response = httpx.Response(200, content=b'audio')

  completed = _consume_audio(response, cast(BinaryIO, _BrokenOutput()), None)

  assert completed is False


def test_finish_player_raises_on_nonzero_when_body_completed() -> None:
  player = cast('subprocess.Popen[bytes]', _FakePlayer(2))

  with pytest.raises(SpeechCliError, match='ffplay exited with status 2'):
    _finish_player(player, body_completed=True)


def test_finish_player_ignores_nonzero_when_body_incomplete() -> None:
  player = cast('subprocess.Popen[bytes]', _FakePlayer(2))

  _finish_player(player, body_completed=False)


def test_finish_player_ignores_negative_return_code_even_if_completed() -> None:
  player = cast('subprocess.Popen[bytes]', _FakePlayer(-2))

  _finish_player(player, body_completed=True)


def test_finish_player_ignores_broken_pipe_on_close() -> None:
  player = cast('subprocess.Popen[bytes]', _FakePlayer(0, stdin=_BrokenStdin()))

  _finish_player(player, body_completed=True)


def test_open_output_writes_to_given_file(tmp_path: Path) -> None:
  output_path = tmp_path / 'speech.mp3'
  options = SpeakOptions('hello', 'http://gateway', output=output_path)

  with ExitStack() as stack:
    output = _open_output(stack, options, _FakeStdout(tty=True))
    output.write(b'audio')

  assert output_path.read_bytes() == b'audio'


def test_open_output_wraps_oserror(tmp_path: Path) -> None:
  bad_path = tmp_path / 'missing-dir' / 'speech.mp3'
  options = SpeakOptions('hello', 'http://gateway', output=bad_path)

  with ExitStack() as stack:
    with pytest.raises(SpeechCliError, match='cannot open output file'):
      _open_output(stack, options, _FakeStdout(tty=True))


def test_open_output_tees_when_play_and_stdout_not_tty() -> None:
  stdout = _FakeStdout(tty=False)
  options = SpeakOptions('hello', 'http://gateway', play=True)

  with ExitStack() as stack:
    output = _open_output(stack, options, stdout)

  assert output is stdout


def test_open_output_uses_devnull_when_play_and_stdout_is_tty() -> None:
  stdout = _FakeStdout(tty=True)
  options = SpeakOptions('hello', 'http://gateway', play=True)

  with ExitStack() as stack:
    output = _open_output(stack, options, stdout)
    output.write(b'x')  # devnull silently accepts writes

  assert output is not stdout


def test_open_output_refuses_no_play_to_tty() -> None:
  stdout = _FakeStdout(tty=True)
  options = SpeakOptions('hello', 'http://gateway', play=False)

  with ExitStack() as stack:
    with pytest.raises(SpeechCliError, match='refusing to write binary audio'):
      _open_output(stack, options, stdout)


def test_open_output_no_play_non_tty_writes_stdout() -> None:
  stdout = _FakeStdout(tty=False)
  options = SpeakOptions('hello', 'http://gateway', play=False)

  with ExitStack() as stack:
    output = _open_output(stack, options, stdout)

  assert output is stdout


def test_open_player_missing_ffplay(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(speech_cli_module.shutil, 'which', lambda _name: None)
  options = SpeakOptions('hello', 'http://gateway', ffplay_path='nonexistent-ffplay')

  with pytest.raises(SpeechCliError, match='ffplay executable not found'):
    _open_player(options)


def test_speak_writes_to_file_end_to_end(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b'audio-bytes')

  transport = httpx.MockTransport(respond)
  real_client = httpx.Client

  def client_factory(*, timeout: httpx.Timeout) -> httpx.Client:
    return real_client(timeout=timeout, transport=transport)

  monkeypatch.setattr(speech_cli_module.httpx, 'Client', client_factory)
  output_path = tmp_path / 'out.mp3'
  options = SpeakOptions('hello', 'http://gateway', output=output_path, play=False)

  speak(options, BytesIO())

  assert output_path.read_bytes() == b'audio-bytes'


def test_speak_wraps_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('connection refused', request=request)

  transport = httpx.MockTransport(respond)
  real_client = httpx.Client

  def client_factory(*, timeout: httpx.Timeout) -> httpx.Client:
    return real_client(timeout=timeout, transport=transport)

  monkeypatch.setattr(speech_cli_module.httpx, 'Client', client_factory)
  options = SpeakOptions('hello', 'http://gateway', play=False)

  with pytest.raises(SpeechCliError, match='gateway request failed'):
    speak(options, BytesIO())


def test_config_formats_active_defaults() -> None:
  result = format_config(_health_report(), 'http://gateway/')

  assert 'Model / engine' in result
  assert 'hexgrad/Kokoro-82M' in result
  assert 'Device' in result
  assert 'mps' in result
  assert 'Default voice' in result
  assert 'af_heart' in result
  assert 'Default speed' in result
  assert '1.0' in result
  assert 'Streaming' not in result
  assert 'Playback' not in result


def test_health_command_reports_gateway(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  monkeypatch.setattr(cli, 'fetch_health', lambda _url, _timeout: _health_report())

  cli.main(['health', '--base-url', 'http://gateway/'])

  assert capsys.readouterr().out == 'healthy  http://gateway\n'


def test_fetch_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
  payload = json.loads(_health_report().model_dump_json(by_alias=True))

  def get(url: str, timeout: float) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request('GET', url))

  monkeypatch.setattr(speech_cli_module.httpx, 'get', get)

  report = fetch_health('http://gateway', 5.0)

  assert report.ok is True
  assert report.primary_engine == 'kokoro'


def test_fetch_health_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
  def raise_error(url: str, timeout: float) -> httpx.Response:
    raise httpx.ConnectError('connection refused')

  monkeypatch.setattr(speech_cli_module.httpx, 'get', raise_error)

  with pytest.raises(SpeechCliError, match='health check failed'):
    fetch_health('http://gateway', 5.0)


def test_fetch_health_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
  def get(url: str, timeout: float) -> httpx.Response:
    return httpx.Response(200, json={'ok': True}, request=httpx.Request('GET', url))

  monkeypatch.setattr(speech_cli_module.httpx, 'get', get)

  with pytest.raises(SpeechCliError, match='health check failed'):
    fetch_health('http://gateway', 5.0)


def test_run_update_requires_uv(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(cli.shutil, 'which', lambda _name: None)

  with pytest.raises(SpeechCliError, match='uv is required'):
    cli._run_update()


def test_run_update_fails_on_nonzero_returncode(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    cli.shutil, 'which', lambda name: '/usr/bin/uv' if name == 'uv' else None
  )
  monkeypatch.setattr(
    cli.subprocess,
    'run',
    lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
  )

  with pytest.raises(SpeechCliError, match='could not update'):
    cli._run_update()


def test_run_update_runs_installed_tts_version(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[list[str]] = []
  paths = {'uv': '/usr/bin/uv', 'tts': '/usr/local/bin/tts'}

  def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
    calls.append(command)
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(cli.shutil, 'which', lambda name: paths.get(name))
  monkeypatch.setattr(cli.subprocess, 'run', run)

  cli._run_update()

  assert calls == [
    [
      '/usr/bin/uv',
      'tool',
      'upgrade',
      'tts-gateway',
      '--reinstall-package',
      'tts-gateway',
    ],
    ['/usr/local/bin/tts', '--version'],
  ]


def test_run_update_skips_version_report_when_tts_not_found(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[list[str]] = []

  def which(name: str) -> str | None:
    return '/usr/bin/uv' if name == 'uv' else None

  def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
    calls.append(command)
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(cli.shutil, 'which', which)
  monkeypatch.setattr(cli.subprocess, 'run', run)

  cli._run_update()

  assert len(calls) == 1


def test_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
  def raise_interrupt(_args: object) -> None:
    raise KeyboardInterrupt

  monkeypatch.setattr(cli, '_run_speak', raise_interrupt)

  with pytest.raises(SystemExit) as excinfo:
    cli.main(['speak', 'hi'])

  assert excinfo.value.code == 130
