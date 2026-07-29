from __future__ import annotations

import subprocess
from io import BytesIO, StringIO
from pathlib import Path

import httpx
import pytest

from tts_gateway import cli
from tts_gateway.speech_cli import (
  HealthReport,
  SpeakOptions,
  SpeechCliError,
  _consume_audio,
  _speech_response,
  format_config,
  read_speak_text,
)


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
  with pytest.raises(SystemExit, match='0'):
    cli.main(['--version'])

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


def test_stream_request_sends_json() -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == '/tts/stream'
    assert request.content == b'{"text":"hello","speed":1.25}'
    return httpx.Response(200, content=b'audio')

  transport = httpx.MockTransport(respond)
  options = SpeakOptions('hello', 'http://gateway', speed=1.25)
  with httpx.Client(transport=transport) as client:
    with _speech_response(client, options) as response:
      assert response.read() == b'audio'


def test_no_play_writes_audio_to_stdout() -> None:
  response = httpx.Response(200, content=b'audio')
  stdout = BytesIO()
  options = SpeakOptions('hello', 'http://gateway', play=False)

  _consume_audio(response, options, stdout)

  assert stdout.getvalue() == b'audio'


def test_output_file_receives_audio(tmp_path: Path) -> None:
  output = tmp_path / 'speech.mp3'
  response = httpx.Response(200, content=b'audio')
  options = SpeakOptions('hello', 'http://gateway', output=output, play=False)

  _consume_audio(response, options, BytesIO())

  assert output.read_bytes() == b'audio'


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
  assert 'Streaming' in result
  assert 'Playback' in result


def test_health_command_reports_gateway(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  monkeypatch.setattr(cli, 'fetch_health', lambda _url, _timeout: _health_report())

  cli.main(['health', '--base-url', 'http://gateway/'])

  assert capsys.readouterr().out == 'healthy  http://gateway\n'


def test_update_uses_uv_tool_upgrade(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  calls: list[list[str]] = []

  def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
    calls.append(command)
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(cli.shutil, 'which', lambda _name: '/usr/bin/uv')
  monkeypatch.setattr(cli.subprocess, 'run', run)
  monkeypatch.setattr(cli, 'package_version', lambda: '1.2.0')

  cli.main(['update'])

  assert calls == [
    [
      '/usr/bin/uv',
      'tool',
      'upgrade',
      'tts-gateway',
      '--reinstall-package',
      'tts-gateway',
    ]
  ]
  assert capsys.readouterr().out == 'tts 1.2.0\n'
