from __future__ import annotations

import json
import subprocess
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
  _encoded_ffplay_args,
  _open_output,
  _pcm_ffplay_args,
  _Player,
  _raise_response_error,
  _resolve_ffplay,
  _speech_response,
  _wants_pcm,
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


class _TrackingOutput(BytesIO):
  """A BytesIO that counts flush() calls, for tee vs. file sink assertions."""

  def __init__(self) -> None:
    super().__init__()
    self.flush_calls = 0

  def flush(self) -> None:
    self.flush_calls += 1
    super().flush()


class _FakeProcess:
  """A minimal stand-in for subprocess.Popen[bytes], for _Player tests."""

  def __init__(self, return_code: int = 0, stdin: Any | None = None) -> None:
    self.returncode = return_code
    self.stdin = stdin if stdin is not None else BytesIO()
    self._return_code = return_code

  def wait(self) -> int:
    return self._return_code


class _BrokenStdin:
  """A file-like object whose writes and close always raise BrokenPipeError."""

  def write(self, data: bytes) -> int:
    raise BrokenPipeError

  def flush(self) -> None:
    pass

  def close(self) -> None:
    raise BrokenPipeError


class _RecordedStdin:
  """A stdin stand-in that keeps written bytes readable even after close()."""

  def __init__(self) -> None:
    self.chunks: list[bytes] = []

  def write(self, data: bytes) -> int:
    self.chunks.append(data)
    return len(data)

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass


class _RecordedPopen:
  """Fake ffplay process that records its argv, for spawn-time assertions."""

  def __init__(self, argv: list[str]) -> None:
    self.argv = argv
    self.stdin = _RecordedStdin()
    self.returncode = 0

  def wait(self) -> int:
    return self.returncode


def _patch_popen(monkeypatch: pytest.MonkeyPatch) -> list[_RecordedPopen]:
  """Replace subprocess.Popen with a recorder; returns the list of spawned processes."""
  processes: list[_RecordedPopen] = []

  def fake_popen(argv: list[str], *, stdin: int) -> _RecordedPopen:
    process = _RecordedPopen(argv)
    processes.append(process)
    return process

  monkeypatch.setattr(speech_cli_module.subprocess, 'Popen', fake_popen)
  return processes


def _patch_which_finds_everything(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    speech_cli_module.shutil, 'which', lambda name: f'/usr/bin/{name}'
  )


def _patch_client(
  monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport
) -> None:
  real_client = httpx.Client

  def client_factory(*, timeout: httpx.Timeout) -> httpx.Client:
    return real_client(timeout=timeout, transport=transport)

  monkeypatch.setattr(speech_cli_module.httpx, 'Client', client_factory)


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


def test_stream_request_sends_json_to_v1_speech_stream() -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == '/v1/speech/stream'
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


def test_stream_request_includes_format_when_requested() -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert json.loads(request.content)['format'] == 'pcm'
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  options = SpeakOptions('hello', 'http://gateway')
  with httpx.Client(transport=transport) as client:
    with _speech_response(client, options, request_format='pcm') as response:
      response.read()


def test_stream_request_omits_format_by_default() -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert 'format' not in json.loads(request.content)
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  options = SpeakOptions('hello', 'http://gateway')
  with httpx.Client(transport=transport) as client:
    with _speech_response(client, options) as response:
      response.read()


def test_nonstream_request_sends_json_to_v1_speech() -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == '/v1/speech'
    assert request.headers['content-type'] == 'application/json'
    assert json.loads(request.content) == {
      'text': 'hello',
      'voice': 'af_heart',
      'speed': 1.25,
    }
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


def test_wants_pcm_true_only_for_play_only_tty_stream_mode() -> None:
  base = SpeakOptions('hello', 'http://gateway')

  assert _wants_pcm(base, _FakeStdout(tty=True)) is True
  assert _wants_pcm(base, _FakeStdout(tty=False)) is False

  no_stream = SpeakOptions('hello', 'http://gateway', stream=False)
  assert _wants_pcm(no_stream, _FakeStdout(tty=True)) is False

  no_play = SpeakOptions('hello', 'http://gateway', play=False)
  assert _wants_pcm(no_play, _FakeStdout(tty=True)) is False

  with_output = SpeakOptions('hello', 'http://gateway', output=Path('out.mp3'))
  assert _wants_pcm(with_output, _FakeStdout(tty=True)) is False


def test_pcm_ffplay_args_uses_response_headers() -> None:
  response = httpx.Response(
    200,
    headers={
      'x-tts-pcm-format': 's16le',
      'x-tts-sample-rate': '24000',
      'x-tts-channels': '1',
    },
  )

  assert _pcm_ffplay_args(response) == (
    '-f',
    's16le',
    '-ar',
    '24000',
    '-ch_layout',
    'mono',
    '-fflags',
    'nobuffer',
  )


def test_pcm_ffplay_args_falls_back_to_defaults_when_headers_missing() -> None:
  response = httpx.Response(200)

  assert _pcm_ffplay_args(response) == (
    '-f',
    's16le',
    '-ar',
    '24000',
    '-ch_layout',
    'mono',
    '-fflags',
    'nobuffer',
  )


def test_encoded_ffplay_args_maps_content_type_to_demuxer() -> None:
  mp3 = httpx.Response(200, headers={'content-type': 'audio/mpeg'})
  wav = httpx.Response(200, headers={'content-type': 'audio/wav; charset=binary'})

  assert _encoded_ffplay_args(mp3) == ('-f', 'mp3', '-fflags', 'nobuffer')
  assert _encoded_ffplay_args(wav) == ('-f', 'wav', '-fflags', 'nobuffer')


def test_encoded_ffplay_args_assumes_mp3_when_content_type_missing() -> None:
  response = httpx.Response(200)

  assert _encoded_ffplay_args(response) == ('-f', 'mp3', '-fflags', 'nobuffer')


def test_encoded_ffplay_args_probes_on_unknown_content_type() -> None:
  response = httpx.Response(200, headers={'content-type': 'audio/ogg'})

  assert _encoded_ffplay_args(response) == ('-fflags', 'nobuffer')


def test_speak_no_stream_wav_gateway_uses_wav_demuxer(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == '/v1/speech'
    return httpx.Response(
      200, content=b'wav-bytes', headers={'content-type': 'audio/wav'}
    )

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  _patch_which_finds_everything(monkeypatch)
  processes = _patch_popen(monkeypatch)

  options = SpeakOptions('hello', 'http://gateway', stream=False)
  speak(options, _FakeStdout(tty=True))

  argv = processes[0].argv
  assert argv[argv.index('-f') + 1] == 'wav'
  assert b''.join(processes[0].stdin.chunks) == b'wav-bytes'


def test_no_play_writes_audio_to_stdout() -> None:
  response = httpx.Response(200, content=b'audio')
  stdout = BytesIO()

  completed = _consume_audio(response, stdout, None, flush_output=False)

  assert completed is True
  assert stdout.getvalue() == b'audio'


def test_output_file_receives_audio(tmp_path: Path) -> None:
  output_path = tmp_path / 'speech.mp3'
  response = httpx.Response(200, content=b'audio')

  with output_path.open('wb') as output:
    completed = _consume_audio(response, output, None, flush_output=False)

  assert completed is True
  assert output_path.read_bytes() == b'audio'


def test_consume_audio_flushes_stdout_tee_sink_per_chunk() -> None:
  response = httpx.Response(200, content=b'audio')
  output = _TrackingOutput()

  completed = _consume_audio(response, output, None, flush_output=True)

  assert completed is True
  assert output.flush_calls >= 1


def test_consume_audio_does_not_flush_file_sink() -> None:
  response = httpx.Response(200, content=b'audio')
  output = _TrackingOutput()

  completed = _consume_audio(response, output, None, flush_output=False)

  assert completed is True
  assert output.flush_calls == 0


def test_consume_audio_stops_cleanly_on_broken_player_pipe() -> None:
  response = httpx.Response(200, content=b'audio')
  process = cast('subprocess.Popen[bytes]', _FakeProcess(stdin=_BrokenStdin()))
  player = _Player(process)

  completed = _consume_audio(response, BytesIO(), player, flush_output=False)

  assert completed is False


def test_consume_audio_stops_cleanly_on_broken_output_pipe() -> None:
  class _BrokenOutput:
    def write(self, data: bytes) -> int:
      raise BrokenPipeError

  response = httpx.Response(200, content=b'audio')

  completed = _consume_audio(
    response, cast(BinaryIO, _BrokenOutput()), None, flush_output=False
  )

  assert completed is False


def test_player_finish_raises_on_nonzero_when_body_completed() -> None:
  player = _Player(cast('subprocess.Popen[bytes]', _FakeProcess(2)))

  with pytest.raises(SpeechCliError, match='ffplay exited with status 2'):
    player.finish(body_completed=True)


def test_player_finish_ignores_nonzero_when_body_incomplete() -> None:
  player = _Player(cast('subprocess.Popen[bytes]', _FakeProcess(2)))

  player.finish(body_completed=False)


def test_player_finish_ignores_negative_return_code_even_if_completed() -> None:
  player = _Player(cast('subprocess.Popen[bytes]', _FakeProcess(-2)))

  player.finish(body_completed=True)


def test_player_finish_ignores_broken_pipe_on_close() -> None:
  process = cast('subprocess.Popen[bytes]', _FakeProcess(0, stdin=_BrokenStdin()))
  player = _Player(process)

  player.finish(body_completed=True)


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


def test_resolve_ffplay_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(speech_cli_module.shutil, 'which', lambda _name: None)

  with pytest.raises(SpeechCliError, match='ffplay executable not found'):
    _resolve_ffplay('nonexistent-ffplay')


def test_resolve_ffplay_returns_resolved_path(monkeypatch: pytest.MonkeyPatch) -> None:
  _patch_which_finds_everything(monkeypatch)

  assert _resolve_ffplay('ffplay') == '/usr/bin/ffplay'


def test_speak_writes_to_file_end_to_end(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == '/v1/speech/stream'
    return httpx.Response(200, content=b'audio-bytes')

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  output_path = tmp_path / 'out.mp3'
  options = SpeakOptions('hello', 'http://gateway', output=output_path, play=False)

  speak(options, BytesIO())

  assert output_path.read_bytes() == b'audio-bytes'


def test_speak_wraps_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('connection refused', request=request)

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  options = SpeakOptions('hello', 'http://gateway', play=False)

  with pytest.raises(SpeechCliError, match='gateway request failed'):
    speak(options, BytesIO())


def test_speak_pcm_play_only_requests_pcm_and_uses_pcm_ffplay_args(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert json.loads(request.content)['format'] == 'pcm'
    return httpx.Response(
      200,
      content=b'pcm-bytes',
      headers={
        'x-tts-pcm-format': 's16le',
        'x-tts-sample-rate': '24000',
        'x-tts-channels': '1',
      },
    )

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  _patch_which_finds_everything(monkeypatch)
  processes = _patch_popen(monkeypatch)

  options = SpeakOptions('hello', 'http://gateway')  # play + stream default, no output
  speak(options, _FakeStdout(tty=True))

  assert len(processes) == 1
  argv = processes[0].argv
  assert argv[0] == '/usr/bin/ffplay'
  assert argv[argv.index('-f') + 1] == 's16le'
  assert argv[argv.index('-ar') + 1] == '24000'
  assert argv[argv.index('-ch_layout') + 1] == 'mono'
  assert argv[argv.index('-fflags') + 1] == 'nobuffer'
  assert argv[-2:] == ['-i', 'pipe:0']
  assert b''.join(processes[0].stdin.chunks) == b'pcm-bytes'


def test_speak_file_output_requests_mp3_and_uses_mp3_ffplay_args(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert 'format' not in json.loads(request.content)
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  _patch_which_finds_everything(monkeypatch)
  processes = _patch_popen(monkeypatch)

  output_path = tmp_path / 'out.mp3'
  options = SpeakOptions('hello', 'http://gateway', output=output_path)
  speak(options, _FakeStdout(tty=True))

  assert len(processes) == 1
  argv = processes[0].argv
  assert argv[argv.index('-f') + 1] == 'mp3'
  assert argv[argv.index('-fflags') + 1] == 'nobuffer'
  assert argv[-2:] == ['-i', 'pipe:0']
  assert output_path.read_bytes() == b'audio'


def test_speak_tee_mode_requests_mp3_and_tees_stdout(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert 'format' not in json.loads(request.content)
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  _patch_which_finds_everything(monkeypatch)
  processes = _patch_popen(monkeypatch)

  stdout = _FakeStdout(tty=False)
  options = SpeakOptions('hello', 'http://gateway')
  speak(options, stdout)

  assert len(processes) == 1
  assert processes[0].argv[processes[0].argv.index('-f') + 1] == 'mp3'
  assert stdout.getvalue() == b'audio'


def test_speak_no_play_omits_format_and_spawns_no_player(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert 'format' not in json.loads(request.content)
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  processes = _patch_popen(monkeypatch)

  stdout = BytesIO()
  options = SpeakOptions('hello', 'http://gateway', play=False)
  speak(options, stdout)

  assert processes == []
  assert stdout.getvalue() == b'audio'


def test_speak_gateway_error_spawns_no_player(monkeypatch: pytest.MonkeyPatch) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, content=b'engine crashed')

  transport = httpx.MockTransport(respond)
  _patch_client(monkeypatch, transport)
  _patch_which_finds_everything(monkeypatch)
  processes = _patch_popen(monkeypatch)

  options = SpeakOptions('hello', 'http://gateway')
  with pytest.raises(SpeechCliError, match='engine crashed'):
    speak(options, _FakeStdout(tty=True))

  assert processes == []


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
