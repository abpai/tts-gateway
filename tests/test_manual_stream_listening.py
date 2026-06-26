"""Tests for scripts/manual_stream_listening.py helpers."""

from __future__ import annotations

import importlib.util
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  'manual_stream_listening',
  ROOT / 'scripts' / 'manual_stream_listening.py',
)
assert SPEC is not None and SPEC.loader is not None
listening = importlib.util.module_from_spec(SPEC)
sys.modules['manual_stream_listening'] = listening
SPEC.loader.exec_module(listening)

transport = listening.transport


def test_default_endpoints() -> None:
  args = listening.parse_args(['--base-url', 'http://127.0.0.1:8000'])
  assert transport.default_endpoints(args) == ['/tts/stream', '/tts/stream/pcm']


def test_default_text_spans_multiple_chunks() -> None:
  assert len(listening.DEFAULT_LISTENING_TEXT) >= 700


def test_resolve_report_path_defaults_under_output_dir() -> None:
  args = listening.parse_args(['--output-dir', '/tmp/listening'])
  assert listening.resolve_report_path(args) == Path(
    '/tmp/listening/listening-report.md'
  )


def test_resolve_report_path_honors_override() -> None:
  args = listening.parse_args(['--report', '/tmp/custom.md'])
  assert listening.resolve_report_path(args) == Path('/tmp/custom.md')


def test_wav_review_path_uses_endpoint_slug() -> None:
  path = listening.wav_review_path(Path('/tmp/out'), '/tts/stream/pcm')
  assert path == Path('/tmp/out/wav_tts_stream_pcm.wav')


def test_wav_replay_command_shell() -> None:
  command = listening.wav_replay_command_shell(
    Path('/tmp/wav_tts_stream.wav'),
    'ffplay',
  )
  assert command == ('ffplay -autoexit -nodisp -loglevel error /tmp/wav_tts_stream.wav')


def _write_test_wav(
  path: Path,
  *,
  samples: np.ndarray,
  sample_rate: int = 24000,
  channels: int = 1,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with wave.open(str(path), 'wb') as wav_writer:
    wav_writer.setnchannels(channels)
    wav_writer.setsampwidth(2)
    wav_writer.setframerate(sample_rate)
    wav_writer.writeframes(samples.astype(np.int16).tobytes())


def test_analyze_wav_reports_metrics_and_pass(tmp_path: Path) -> None:
  samples = np.array([0, 1000, -2000, 500, -500] * 2400, dtype=np.int16)
  wav_path = tmp_path / 'sample.wav'
  _write_test_wav(wav_path, samples=samples)
  summary = listening.analyze_wav(wav_path)
  assert summary.status == 'PASS'
  assert summary.sample_rate == 24000
  assert summary.channels == 1
  assert summary.sample_width == 2
  assert summary.peak_abs == 2000
  assert summary.rms > 0
  assert 0.49 <= summary.duration_s <= 0.51


def test_analyze_wav_warns_on_all_silent(tmp_path: Path) -> None:
  wav_path = tmp_path / 'silent.wav'
  _write_test_wav(wav_path, samples=np.zeros(4800, dtype=np.int16))
  summary = listening.analyze_wav(wav_path)
  assert summary.status == 'WARN'
  assert 'all_silent' in summary.warnings


def test_analyze_wav_warns_on_empty_audio(tmp_path: Path) -> None:
  wav_path = tmp_path / 'empty.wav'
  _write_test_wav(wav_path, samples=np.array([], dtype=np.int16))
  summary = listening.analyze_wav(wav_path)
  assert summary.status == 'WARN'
  assert 'no_samples' in summary.warnings


def test_analyze_wav_warns_on_large_adjacent_jump(tmp_path: Path) -> None:
  samples = np.zeros(100, dtype=np.int16)
  samples[50] = 30000
  wav_path = tmp_path / 'jump.wav'
  _write_test_wav(wav_path, samples=samples)
  summary = listening.analyze_wav(wav_path)
  assert summary.status == 'WARN'
  assert 'large_adjacent_jump' in summary.warnings


def test_replay_command_shell_pcm_includes_format_args() -> None:
  command = listening.replay_command_shell(
    payload_path=Path('/tmp/pcm_tts_stream_pcm.bin'),
    endpoint='/tts/stream/pcm',
    content_type='audio/raw',
    headers={
      'x-tts-sample-rate': '24000',
      'x-tts-channels': '1',
      'x-tts-pcm-format': 's16le',
    },
    ffplay_path='ffplay',
  )
  assert command == (
    'ffplay -autoexit -nodisp -loglevel error -f s16le -ar 24000 -ac 1 '
    '-i /tmp/pcm_tts_stream_pcm.bin'
  )


def test_replay_command_shell_mp3() -> None:
  command = listening.replay_command_shell(
    payload_path=Path('/tmp/mp3_tts_stream.bin'),
    endpoint='/tts/stream',
    content_type='audio/mpeg',
    headers={},
    ffplay_path='/opt/bin/ffplay',
  )
  assert command == (
    '/opt/bin/ffplay -autoexit -nodisp -loglevel error -i /tmp/mp3_tts_stream.bin'
  )


def test_render_report_includes_limitation_and_pending_verdict() -> None:
  fetch = transport.StreamFetchResult(
    endpoint='/tts/stream/pcm',
    status=200,
    content_type='audio/raw',
    headers={
      'x-tts-sample-rate': '24000',
      'x-tts-channels': '1',
      'x-tts-pcm-format': 's16le',
    },
    payload=b'\x00\x00',
  )
  result = listening.ListeningRunResult(
    endpoint='/tts/stream/pcm',
    fetch=fetch,
    decode=transport.DecodeCheck(
      endpoint='/tts/stream/pcm',
      ok=True,
      output_path=Path('/tmp/pcm_tts_stream_pcm.bin'),
    ),
    wav_review=listening.WavReview(
      path=Path('/tmp/wav_tts_stream_pcm.wav'),
      waveform=listening.WaveformSummary(
        duration_s=1.0,
        sample_rate=24000,
        channels=1,
        sample_width=2,
        peak_abs=5000,
        rms=1200.0,
        longest_silence_s=0.01,
        max_adjacent_jump=500,
        status='PASS',
      ),
    ),
  )
  report = listening.render_report(
    base_url='http://127.0.0.1:45123',
    text='sample text',
    health={
      'ok': True,
      'primaryEngine': 'kokoro',
      'fallbackEngine': None,
      'engineChain': ['kokoro'],
      'streamFirstChunkMaxChars': 180,
      'streamChunkMaxChars': 500,
    },
    results=[result],
    ffplay_path='ffplay',
    generated_at=datetime(2026, 6, 26, tzinfo=UTC),
  )
  assert 'do **not** validate perceived audio quality' in report
  assert 'waveform sanity' in report.lower()
  assert '- wav review: `/tmp/wav_tts_stream_pcm.wav`' in report
  assert (
    'ffplay -autoexit -nodisp -loglevel error /tmp/wav_tts_stream_pcm.wav' in report
  )
  assert 'raw replay (fallback):' in report
  assert '**Status:** PENDING' in report
  assert '- [ ] No clicks or pops at chunk boundaries' in report
  assert '- [ ] Option+R replay on selected text sounds correct' in report
  assert 'primaryEngine: `kokoro`' in report
  assert '-f s16le -ar 24000 -ac 1' in report
  assert 'PASSED' not in report


def test_render_report_notes_unavailable_health() -> None:
  report = listening.render_report(
    base_url='http://127.0.0.1:45123',
    text='x',
    health={'ok': False, 'error': 'ConnectionRefusedError: refused'},
    results=[],
    ffplay_path='ffplay',
    generated_at=datetime(2026, 6, 26, tzinfo=UTC),
  )
  assert 'unavailable: ConnectionRefusedError: refused' in report


def test_play_payload_invokes_ffplay() -> None:
  with patch('manual_stream_listening.subprocess.run') as run:
    run.return_value.returncode = 0
    run.return_value.stderr = b''
    listening.play_payload(
      payload_path=Path('/tmp/pcm.bin'),
      endpoint='/tts/stream/pcm',
      content_type='audio/raw',
      headers={
        'x-tts-sample-rate': '24000',
        'x-tts-channels': '1',
        'x-tts-pcm-format': 's16le',
      },
      ffplay_path='ffplay',
    )
  command = run.call_args.args[0]
  assert command[:6] == ['ffplay', '-autoexit', '-nodisp', '-loglevel', 'error', '-f']
  assert command[-1] == '/tmp/pcm.bin'


def test_play_payload_prefers_wav_review_file(tmp_path: Path) -> None:
  wav_path = tmp_path / 'wav_tts_stream_pcm.wav'
  wav_path.write_bytes(b'RIFF')
  with patch('manual_stream_listening.subprocess.run') as run:
    run.return_value.returncode = 0
    run.return_value.stderr = b''
    listening.play_payload(
      payload_path=Path('/tmp/pcm.bin'),
      endpoint='/tts/stream/pcm',
      content_type='audio/raw',
      headers={
        'x-tts-sample-rate': '24000',
        'x-tts-channels': '1',
        'x-tts-pcm-format': 's16le',
      },
      ffplay_path='ffplay',
      wav_path=wav_path,
    )
  command = run.call_args.args[0]
  assert command == [
    'ffplay',
    '-autoexit',
    '-nodisp',
    '-loglevel',
    'error',
    str(wav_path),
  ]


def test_run_workflow_records_fetch_error(tmp_path: Path) -> None:
  args = listening.parse_args(['--output-dir', str(tmp_path)])

  with patch(
    'manual_stream_listening.fetch_health',
    return_value={'ok': False, 'error': 'refused'},
  ):
    with patch(
      'manual_stream_listening.transport.fetch_stream',
      side_effect=OSError('refused'),
    ):
      health, results = listening.run_workflow(args)

  assert health == {'ok': False, 'error': 'refused'}
  assert len(results) == 2
  assert all(result.decode.ok is False for result in results)
  assert all('OSError: refused' == result.decode.error for result in results)
  assert all(
    result.wav_review is not None
    and result.wav_review.waveform is not None
    and result.wav_review.waveform.status == 'SKIP'
    for result in results
  )


def test_run_workflow_builds_wav_review_on_successful_decode(tmp_path: Path) -> None:
  args = listening.parse_args(['--output-dir', str(tmp_path)])
  fetch = transport.StreamFetchResult(
    endpoint='/tts/stream',
    status=200,
    content_type='audio/mpeg',
    headers={},
    payload=b'fake-mp3',
  )
  payload_path = tmp_path / 'mp3_tts_stream.bin'
  payload_path.write_bytes(fetch.payload)
  decode = transport.DecodeCheck(
    endpoint='/tts/stream',
    ok=True,
    output_path=payload_path,
  )
  wav_path = tmp_path / 'wav_tts_stream.wav'
  waveform = listening.WaveformSummary(
    duration_s=2.0,
    sample_rate=24000,
    channels=1,
    sample_width=2,
    peak_abs=9000,
    rms=1500.0,
    longest_silence_s=0.02,
    max_adjacent_jump=800,
    status='PASS',
  )

  with patch(
    'manual_stream_listening.fetch_health',
    return_value={'ok': True},
  ):
    with patch(
      'manual_stream_listening.transport.fetch_stream',
      return_value=fetch,
    ):
      with patch(
        'manual_stream_listening.transport.validate_fetch',
        return_value=decode,
      ):
        with patch(
          'manual_stream_listening.build_wav_review',
          return_value=listening.WavReview(path=wav_path, waveform=waveform),
        ) as build_wav:
          health, results = listening.run_workflow(args)

  assert health == {'ok': True}
  assert len(results) == 2
  assert build_wav.call_count == 2
  assert results[0].wav_review is not None
  assert results[0].wav_review.path == wav_path
  assert results[0].wav_review.waveform == waveform


def _pass_listening_result() -> listening.ListeningRunResult:
  return listening.ListeningRunResult(
    endpoint='/tts/stream',
    fetch=transport.StreamFetchResult('/tts/stream', 200, 'audio/mpeg', {}, b'x'),
    decode=transport.DecodeCheck('/tts/stream', ok=True, output_path=Path('x.bin')),
    wav_review=listening.WavReview(
      path=Path('x.wav'),
      waveform=listening.WaveformSummary(
        duration_s=1.0,
        sample_rate=24000,
        channels=1,
        sample_width=2,
        peak_abs=1000,
        rms=500.0,
        longest_silence_s=0.01,
        max_adjacent_jump=100,
        status='PASS',
      ),
    ),
  )


def test_workflow_failed_on_waveform_warn_or_skip() -> None:
  ok = _pass_listening_result()
  warn = listening.ListeningRunResult(
    endpoint='/tts/stream/pcm',
    fetch=transport.StreamFetchResult('/tts/stream/pcm', 200, 'audio/raw', {}, b'x'),
    decode=transport.DecodeCheck('/tts/stream/pcm', ok=True, output_path=Path('x.bin')),
    wav_review=listening.WavReview(
      path=Path('x.wav'),
      waveform=listening.WaveformSummary(
        duration_s=1.0,
        sample_rate=24000,
        channels=1,
        sample_width=2,
        peak_abs=0,
        rms=0.0,
        longest_silence_s=0.0,
        max_adjacent_jump=0,
        status='WARN',
        warnings=('all_silent',),
      ),
    ),
  )
  skip = listening.ListeningRunResult(
    endpoint='/tts/stream',
    fetch=transport.StreamFetchResult('/tts/stream', 200, 'audio/mpeg', {}, b'x'),
    decode=transport.DecodeCheck('/tts/stream', ok=True, output_path=Path('x.bin')),
    wav_review=listening.WavReview(
      path=None,
      error='ffmpeg wav export failed',
      waveform=listening.WaveformSummary.skipped(),
    ),
  )
  assert listening.workflow_failed([ok]) is False
  assert listening.workflow_failed([ok, warn]) is True
  assert listening.workflow_failed([ok, skip]) is True


def test_workflow_failed_on_decode_or_play_errors() -> None:
  ok = _pass_listening_result()
  bad_decode = listening.ListeningRunResult(
    endpoint='/tts/stream',
    fetch=transport.StreamFetchResult('/tts/stream', 503, None, {}, b''),
    decode=transport.DecodeCheck(
      '/tts/stream', ok=False, output_path=None, error='HTTP 503'
    ),
  )
  bad_play = listening.ListeningRunResult(
    endpoint='/tts/stream/pcm',
    fetch=transport.StreamFetchResult('/tts/stream/pcm', 200, 'audio/raw', {}, b'x'),
    decode=transport.DecodeCheck('/tts/stream/pcm', ok=True, output_path=Path('x.bin')),
    wav_review=ok.wav_review,
    play_ok=False,
    play_error='ffplay failed',
  )
  bad_wav = listening.ListeningRunResult(
    endpoint='/tts/stream/pcm',
    fetch=transport.StreamFetchResult('/tts/stream/pcm', 200, 'audio/raw', {}, b'x'),
    decode=transport.DecodeCheck('/tts/stream/pcm', ok=True, output_path=Path('x.bin')),
    wav_review=listening.WavReview(error='ffmpeg wav export failed'),
  )
  assert listening.workflow_failed([ok]) is False
  assert listening.workflow_failed([ok, bad_decode]) is True
  assert listening.workflow_failed([ok, bad_play]) is True
  assert listening.workflow_failed([ok, bad_wav]) is True


def test_main_returns_nonzero_on_failure(tmp_path: Path) -> None:
  bad = listening.ListeningRunResult(
    endpoint='/tts/stream',
    fetch=transport.StreamFetchResult('/tts/stream', 503, None, {}, b''),
    decode=transport.DecodeCheck(
      '/tts/stream', ok=False, output_path=None, error='HTTP 503'
    ),
  )
  with patch(
    'manual_stream_listening.run_workflow',
    return_value=({'ok': True}, [bad]),
  ):
    assert (
      listening.main(
        ['--output-dir', str(tmp_path), '--report', str(tmp_path / 'r.md')],
      )
      == 1
    )
  assert (tmp_path / 'r.md').exists()


def test_main_returns_zero_and_writes_report(tmp_path: Path) -> None:
  ok = listening.ListeningRunResult(
    endpoint='/tts/stream',
    fetch=transport.StreamFetchResult('/tts/stream', 200, 'audio/mpeg', {}, b'x'),
    decode=transport.DecodeCheck('/tts/stream', ok=True, output_path=Path('x.bin')),
    wav_review=listening.WavReview(
      path=Path('x.wav'),
      waveform=listening.WaveformSummary(
        duration_s=1.0,
        sample_rate=24000,
        channels=1,
        sample_width=2,
        peak_abs=1000,
        rms=500.0,
        longest_silence_s=0.01,
        max_adjacent_jump=100,
        status='PASS',
      ),
    ),
  )
  with patch(
    'manual_stream_listening.run_workflow',
    return_value=({'ok': True, 'primaryEngine': 'kokoro'}, [ok]),
  ):
    assert (
      listening.main(
        ['--output-dir', str(tmp_path), '--report', str(tmp_path / 'r.md')],
      )
      == 0
    )
  report = (tmp_path / 'r.md').read_text()
  assert 'PENDING' in report
