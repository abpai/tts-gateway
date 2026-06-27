#!/usr/bin/env python3
"""Manual listening smoke workflow for stream endpoints."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
  from scripts import check_stream_transport as transport
  from scripts._gateway_http import NETWORK_ERRORS, fetch_health
except ModuleNotFoundError:
  import check_stream_transport as transport  # type: ignore[no-redef]
  from _gateway_http import (  # type: ignore[no-redef]
    NETWORK_ERRORS,
    fetch_health,
  )

DEFAULT_LISTENING_TEXT = (
  'First sentence for manual listening validation. '
  'Second sentence exercises the stream chunk boundary. '
  'Third sentence keeps playback going across multiple chunks. '
  'Fourth sentence adds enough material for a third chunk when '
  'stream chunking is active. Fifth sentence confirms pacing stays '
  'natural through the entire sample. '
  'This paragraph is long enough to span multiple stream chunks when '
  'chunking is enabled on the gateway. '
  'Each additional sentence increases the chance that chunk joins, '
  'silence gaps, and pacing issues become audible during playback. '
  'Repeat this material until the sample clearly exceeds three chunks. '
  'This is a longer paragraph selected for listening smoke tests. It has '
  'enough words to exercise chunking and make chunk-boundary artifacts '
  'audible if they exist. ' * 6
).strip()

HUMAN_CHECKLIST = (
  'No clicks or pops at chunk boundaries',
  'No audible gaps or dropped audio between chunks',
  'Playback starts promptly (low time-to-first-audio)',
  'Stop/cancel works in Raycast during streaming',
  'Option+R replay on selected text sounds correct',
)

NEAR_SILENCE_ABS_INT16 = 64
NEAR_EMPTY_DURATION_S = 0.05
NEAR_EMPTY_RMS = 1.0
ALL_SILENT_PEAK_INT16 = NEAR_SILENCE_ABS_INT16
CLIPPING_PEAK_INT16 = 32000
LONG_INTERNAL_SILENCE_S = 3.0
LARGE_ADJACENT_JUMP_INT16 = 25000


@dataclass(frozen=True)
class WaveformSummary:
  duration_s: float
  sample_rate: int
  channels: int
  sample_width: int
  peak_abs: int
  rms: float
  longest_silence_s: float
  max_adjacent_jump: int
  status: str
  warnings: tuple[str, ...] = ()

  @classmethod
  def skipped(cls) -> WaveformSummary:
    return cls(
      duration_s=0.0,
      sample_rate=0,
      channels=0,
      sample_width=0,
      peak_abs=0,
      rms=0.0,
      longest_silence_s=0.0,
      max_adjacent_jump=0,
      status='SKIP',
    )


@dataclass(frozen=True)
class WavReview:
  path: Path | None = None
  error: str | None = None
  waveform: WaveformSummary | None = None


@dataclass(frozen=True)
class ListeningRunResult:
  endpoint: str
  fetch: transport.StreamFetchResult
  decode: transport.DecodeCheck
  wav_review: WavReview | None = None
  play_ok: bool | None = None
  play_error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--base-url', default='http://127.0.0.1:45123')
  parser.add_argument(
    '--endpoint',
    action='append',
    dest='endpoints',
    help='Stream endpoint to validate (default: /tts/stream and /tts/stream/pcm)',
  )
  parser.add_argument('--text', default=DEFAULT_LISTENING_TEXT)
  parser.add_argument('--output-dir', type=Path, default=Path('tmp/stream-listening'))
  parser.add_argument('--ffmpeg-path', default='ffmpeg')
  parser.add_argument('--ffplay-path', default='ffplay')
  parser.add_argument(
    '--play',
    action='store_true',
    help='Play each fetched payload through ffplay after decode validation',
  )
  parser.add_argument(
    '--report',
    type=Path,
    help='Markdown report path (default: <output-dir>/listening-report.md)',
  )
  return parser.parse_args(argv)


def resolve_report_path(args: argparse.Namespace) -> Path:
  if args.report is not None:
    return args.report
  return args.output_dir / 'listening-report.md'


def replay_command_argv(
  *,
  payload_path: Path,
  endpoint: str,
  content_type: str | None,
  headers: dict[str, str],
  ffplay_path: str,
) -> list[str]:
  if endpoint.endswith('/pcm') or content_type == 'audio/raw':
    sample_rate = headers.get('x-tts-sample-rate', '24000')
    channels = headers.get('x-tts-channels', '1')
    pcm_format = headers.get('x-tts-pcm-format', 's16le')
    return [
      ffplay_path,
      '-autoexit',
      '-nodisp',
      '-loglevel',
      'error',
      '-f',
      pcm_format,
      '-ar',
      sample_rate,
      '-ac',
      channels,
      '-i',
      str(payload_path),
    ]
  return [
    ffplay_path,
    '-autoexit',
    '-nodisp',
    '-loglevel',
    'error',
    '-i',
    str(payload_path),
  ]


def replay_command_shell(
  *,
  payload_path: Path,
  endpoint: str,
  content_type: str | None,
  headers: dict[str, str],
  ffplay_path: str,
) -> str:
  command = replay_command_argv(
    payload_path=payload_path,
    endpoint=endpoint,
    content_type=content_type,
    headers=headers,
    ffplay_path=ffplay_path,
  )
  return ' '.join(shlex.quote(part) for part in command)


def wav_review_path(output_dir: Path, endpoint: str) -> Path:
  return output_dir / f'wav{endpoint.replace("/", "_")}.wav'


def wav_replay_command_argv(wav_path: Path, ffplay_path: str) -> list[str]:
  return [
    ffplay_path,
    '-autoexit',
    '-nodisp',
    '-loglevel',
    'error',
    str(wav_path),
  ]


def wav_replay_command_shell(wav_path: Path, ffplay_path: str) -> str:
  command = wav_replay_command_argv(wav_path, ffplay_path)
  return ' '.join(shlex.quote(part) for part in command)


def export_wav_review(
  *,
  payload_path: Path,
  wav_path: Path,
  endpoint: str,
  content_type: str | None,
  headers: dict[str, str],
  ffmpeg_path: str,
) -> None:
  wav_path.parent.mkdir(parents=True, exist_ok=True)
  if endpoint.endswith('/pcm') or content_type == 'audio/raw':
    command = [
      ffmpeg_path,
      '-hide_banner',
      '-loglevel',
      'error',
      '-y',
      '-f',
      headers.get('x-tts-pcm-format', 's16le'),
      '-ar',
      headers.get('x-tts-sample-rate', '24000'),
      '-ac',
      headers.get('x-tts-channels', '1'),
      '-i',
      str(payload_path),
      str(wav_path),
    ]
  else:
    command = [
      ffmpeg_path,
      '-hide_banner',
      '-loglevel',
      'error',
      '-y',
      '-i',
      str(payload_path),
      str(wav_path),
    ]
  process = subprocess.run(command, capture_output=True, check=False)
  if process.returncode != 0:
    stderr = (
      process.stderr.decode(errors='replace').strip() or 'ffmpeg wav export failed'
    )
    raise RuntimeError(stderr)


def _read_wav_samples(
  wav_path: Path,
) -> tuple[np.ndarray, int, int, int, float]:
  with wave.open(str(wav_path), 'rb') as wav_reader:
    sample_rate = wav_reader.getframerate()
    channels = wav_reader.getnchannels()
    sample_width = wav_reader.getsampwidth()
    frame_count = wav_reader.getnframes()
    raw = wav_reader.readframes(frame_count)
  duration_s = frame_count / sample_rate if sample_rate else 0.0
  if sample_width == 2:
    samples = np.frombuffer(raw, dtype=np.int16)
  elif sample_width == 1:
    samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128) * 256
    samples = samples.astype(np.int16)
  elif sample_width == 4:
    samples = (np.frombuffer(raw, dtype=np.int32) / 65536).astype(np.int16)
  else:
    raise ValueError(f'unsupported sample width: {sample_width}')
  return samples, sample_rate, channels, sample_width, duration_s


def _longest_true_run(flags: np.ndarray) -> int:
  if flags.size == 0:
    return 0
  padded = np.concatenate(([False], flags, [False]))
  changes = np.diff(padded.astype(np.int8))
  starts = np.where(changes == 1)[0]
  ends = np.where(changes == -1)[0]
  if starts.size == 0:
    return 0
  return int(np.max(ends - starts))


def analyze_wav(wav_path: Path) -> WaveformSummary:
  samples, sample_rate, channels, sample_width, duration_s = _read_wav_samples(wav_path)
  if samples.size == 0:
    return WaveformSummary(
      duration_s=duration_s,
      sample_rate=sample_rate,
      channels=channels,
      sample_width=sample_width,
      peak_abs=0,
      rms=0.0,
      longest_silence_s=0.0,
      max_adjacent_jump=0,
      status='WARN',
      warnings=('no_samples',),
    )
  sample_f64 = samples.astype(np.float64)
  peak_abs = int(np.max(np.abs(samples.astype(np.int32))))
  rms = float(np.sqrt(np.mean(sample_f64 * sample_f64)))
  silent = np.abs(samples.astype(np.int32)) <= NEAR_SILENCE_ABS_INT16
  longest_silence_samples = _longest_true_run(silent)
  per_second = sample_rate * channels if sample_rate and channels else 1
  longest_silence_s = longest_silence_samples / per_second
  if samples.size < 2:
    max_adjacent_jump = 0
  else:
    max_adjacent_jump = int(np.max(np.abs(np.diff(samples.astype(np.int32)))))

  warnings: list[str] = []
  if duration_s < NEAR_EMPTY_DURATION_S:
    warnings.append('near_empty_duration')
  if rms < NEAR_EMPTY_RMS:
    warnings.append('near_empty_rms')
  if peak_abs <= ALL_SILENT_PEAK_INT16:
    warnings.append('all_silent')
  if peak_abs >= CLIPPING_PEAK_INT16:
    warnings.append('extreme_clipping')
  if longest_silence_s >= LONG_INTERNAL_SILENCE_S:
    warnings.append('long_internal_silence')
  if max_adjacent_jump >= LARGE_ADJACENT_JUMP_INT16:
    warnings.append('large_adjacent_jump')

  return WaveformSummary(
    duration_s=duration_s,
    sample_rate=sample_rate,
    channels=channels,
    sample_width=sample_width,
    peak_abs=peak_abs,
    rms=rms,
    longest_silence_s=longest_silence_s,
    max_adjacent_jump=max_adjacent_jump,
    status='WARN' if warnings else 'PASS',
    warnings=tuple(warnings),
  )


def build_wav_review(
  *,
  payload_path: Path,
  output_dir: Path,
  endpoint: str,
  content_type: str | None,
  headers: dict[str, str],
  ffmpeg_path: str,
) -> WavReview:
  wav_path = wav_review_path(output_dir, endpoint)
  try:
    export_wav_review(
      payload_path=payload_path,
      wav_path=wav_path,
      endpoint=endpoint,
      content_type=content_type,
      headers=headers,
      ffmpeg_path=ffmpeg_path,
    )
    waveform = analyze_wav(wav_path)
  except (RuntimeError, ValueError) as exc:
    return WavReview(path=None, error=str(exc), waveform=WaveformSummary.skipped())
  return WavReview(path=wav_path, waveform=waveform)


def play_payload(
  *,
  payload_path: Path,
  endpoint: str,
  content_type: str | None,
  headers: dict[str, str],
  ffplay_path: str,
  wav_path: Path | None = None,
) -> None:
  if wav_path is not None and wav_path.exists():
    command = wav_replay_command_argv(wav_path, ffplay_path)
  else:
    command = replay_command_argv(
      payload_path=payload_path,
      endpoint=endpoint,
      content_type=content_type,
      headers=headers,
      ffplay_path=ffplay_path,
    )
  process = subprocess.run(command, capture_output=True, check=False)
  if process.returncode != 0:
    stderr = process.stderr.decode(errors='replace').strip() or 'ffplay failed'
    raise RuntimeError(stderr)


def format_health_section(health: dict[str, Any]) -> list[str]:
  lines = ['## Gateway health', '']
  if not health.get('ok'):
    if 'error' in health:
      lines.append(f'- unavailable: {health["error"]}')
    elif 'status' in health:
      lines.append(f'- unavailable: HTTP {health["status"]}')
    else:
      lines.append('- unavailable')
    lines.append('')
    return lines

  lines.extend(
    [
      f'- primaryEngine: `{health.get("primaryEngine")}`',
      f'- fallbackEngine: `{health.get("fallbackEngine")}`',
      f'- engineChain: `{health.get("engineChain")}`',
      (f'- streamFirstChunkMaxChars: `{health.get("streamFirstChunkMaxChars")}`'),
      f'- streamChunkMaxChars: `{health.get("streamChunkMaxChars")}`',
      '',
    ]
  )
  return lines


def format_waveform_section(waveform: WaveformSummary | None) -> list[str]:
  lines = ['', '### Waveform sanity', '']
  if waveform is None or waveform.status == 'SKIP':
    lines.append('- status: SKIP')
    lines.append('')
    return lines
  lines.append(f'- status: {waveform.status}')
  lines.append(f'- duration: {waveform.duration_s:.3f}s')
  lines.append(f'- sampleRate: {waveform.sample_rate}')
  lines.append(f'- channels: {waveform.channels}')
  lines.append(f'- sampleWidth: {waveform.sample_width}')
  lines.append(f'- peakAbs: {waveform.peak_abs}')
  lines.append(f'- rms: {waveform.rms:.2f}')
  lines.append(f'- longestSilence: {waveform.longest_silence_s:.3f}s')
  lines.append(f'- maxAdjacentJump: {waveform.max_adjacent_jump}')
  if waveform.warnings:
    lines.append(f'- warnings: {", ".join(waveform.warnings)}')
  lines.append('')
  return lines


def format_endpoint_section(
  result: ListeningRunResult,
  *,
  ffplay_path: str,
) -> list[str]:
  lines = [f'## `{result.endpoint}`', '', '### Automated checks', '']
  lines.append(f'- HTTP status: {result.fetch.status}')
  decode_label = 'PASS' if result.decode.ok else 'FAIL'
  lines.append(f'- ffmpeg decode: {decode_label}')
  if result.decode.error:
    lines.append(f'- decode error: {result.decode.error}')
  if result.decode.output_path is not None:
    lines.append(f'- payload: `{result.decode.output_path}`')
    replay_command = replay_command_shell(
      payload_path=result.decode.output_path,
      endpoint=result.endpoint,
      content_type=result.fetch.content_type,
      headers=result.fetch.headers,
      ffplay_path=ffplay_path,
    )
    lines.append(f'- raw replay (fallback): `{replay_command}`')
  wav_review = result.wav_review
  if wav_review is not None and wav_review.path is not None:
    lines.append(f'- wav review: `{wav_review.path}`')
    lines.append(
      f'- replay: `{wav_replay_command_shell(wav_review.path, ffplay_path)}`'
    )
  elif wav_review is not None and wav_review.error:
    lines.append(f'- wav review error: {wav_review.error}')
  if wav_review is not None:
    lines.extend(format_waveform_section(wav_review.waveform))
  if result.play_ok is not None:
    play_label = 'PASS' if result.play_ok else 'FAIL'
    lines.append(f'- ffplay playback: {play_label}')
    if result.play_error:
      lines.append(f'- playback error: {result.play_error}')

  lines.extend(['', '### Human listening verdict', ''])
  lines.append('**Status:** PENDING — fill in after listening')
  lines.append('')
  for item in HUMAN_CHECKLIST:
    lines.append(f'- [ ] {item}')
  lines.append('')
  return lines


def render_report(
  *,
  base_url: str,
  text: str,
  health: dict[str, Any],
  results: list[ListeningRunResult],
  ffplay_path: str,
  generated_at: datetime,
) -> str:
  lines = [
    '# Stream Manual Listening Report',
    '',
    f'- generatedAt: `{generated_at.isoformat()}`',
    f'- baseUrl: `{base_url}`',
    f'- textChars: {len(text)}',
    '',
    (
      '> **Limitation:** Automated fetch, ffmpeg decode, and waveform sanity '
      'checks below do **not** validate perceived audio quality. Only a human '
      'listener can complete the checklist in this report.'
    ),
    '',
  ]
  lines.extend(format_health_section(health))
  for result in results:
    lines.extend(
      format_endpoint_section(result, ffplay_path=ffplay_path),
    )
  lines.extend(
    [
      '## Overall human verdict',
      '',
      (
        '**Status:** PENDING — review each endpoint with the replay commands '
        'above, then check items in Raycast (stop/cancel and Option+R).'
      ),
      '',
    ]
  )
  return '\n'.join(lines)


def run_workflow(
  args: argparse.Namespace,
) -> tuple[dict[str, Any], list[ListeningRunResult]]:
  health = fetch_health(args.base_url)
  results: list[ListeningRunResult] = []
  for endpoint in transport.default_endpoints(args):
    try:
      fetch = transport.fetch_stream(args.base_url, endpoint, args.text)
    except NETWORK_ERRORS as exc:
      fetch = transport.StreamFetchResult(
        endpoint=endpoint,
        status=0,
        content_type=None,
        headers={},
        payload=b'',
      )
      decode = transport.DecodeCheck(
        endpoint=endpoint,
        ok=False,
        output_path=None,
        error=f'{type(exc).__name__}: {exc}',
      )
      results.append(
        ListeningRunResult(
          endpoint=endpoint,
          fetch=fetch,
          decode=decode,
          wav_review=WavReview(waveform=WaveformSummary.skipped()),
        )
      )
      continue

    decode = transport.validate_fetch(
      fetch,
      output_dir=args.output_dir,
      ffmpeg_path=args.ffmpeg_path,
    )
    wav_review: WavReview | None = None
    if decode.ok and decode.output_path is not None:
      wav_review = build_wav_review(
        payload_path=decode.output_path,
        output_dir=args.output_dir,
        endpoint=endpoint,
        content_type=fetch.content_type,
        headers=fetch.headers,
        ffmpeg_path=args.ffmpeg_path,
      )
    play_ok: bool | None = None
    play_error: str | None = None
    if args.play and decode.ok and decode.output_path is not None:
      try:
        play_payload(
          payload_path=decode.output_path,
          endpoint=endpoint,
          content_type=fetch.content_type,
          headers=fetch.headers,
          ffplay_path=args.ffplay_path,
          wav_path=wav_review.path if wav_review is not None else None,
        )
        play_ok = True
      except RuntimeError as exc:
        play_ok = False
        play_error = str(exc)
    results.append(
      ListeningRunResult(
        endpoint=endpoint,
        fetch=fetch,
        decode=decode,
        wav_review=wav_review,
        play_ok=play_ok,
        play_error=play_error,
      )
    )
  return health, results


def workflow_failed(results: list[ListeningRunResult]) -> bool:
  for result in results:
    if not result.decode.ok:
      return True
    if result.wav_review is None:
      return True
    if result.wav_review.error:
      return True
    waveform = result.wav_review.waveform
    if waveform is not None and waveform.status != 'PASS':
      return True
    if result.play_ok is False:
      return True
  return False


def write_report(report_path: Path, content: str) -> None:
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(content)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  report_path = resolve_report_path(args)
  health, results = run_workflow(args)
  report = render_report(
    base_url=args.base_url,
    text=args.text,
    health=health,
    results=results,
    ffplay_path=args.ffplay_path,
    generated_at=datetime.now(UTC),
  )
  write_report(report_path, report)
  print(f'Report: {report_path}')
  for result in results:
    if result.decode.ok:
      detail = str(result.decode.output_path)
      if result.wav_review is not None and result.wav_review.path is not None:
        detail = f'{detail}, {result.wav_review.path}'
      print(f'OK  {result.endpoint} -> {detail}')
    else:
      print(f'FAIL {result.endpoint}: {result.decode.error}', file=sys.stderr)
    if result.play_ok is False:
      print(
        f'FAIL {result.endpoint} playback: {result.play_error}',
        file=sys.stderr,
      )
  return 1 if workflow_failed(results) else 0


if __name__ == '__main__':
  raise SystemExit(main())
