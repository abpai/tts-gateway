from __future__ import annotations

import subprocess
import wave
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from tts_gateway.config import OutputFormat
from tts_gateway.engines.base import AudioChunk

SAMPLE_FORMAT_BY_WIDTH = {
  1: 'u8',
  2: 's16',
  4: 's32',
}


def wav_bytes_to_chunk(payload: bytes) -> AudioChunk:
  with wave.open(BytesIO(payload), 'rb') as wav_reader:
    return AudioChunk(
      pcm_bytes=wav_reader.readframes(wav_reader.getnframes()),
      sample_rate=wav_reader.getframerate(),
      channels=wav_reader.getnchannels(),
      sample_width=wav_reader.getsampwidth(),
    )


def chunk_to_wav_bytes(chunk: AudioChunk) -> bytes:
  output = BytesIO()
  with wave.open(output, 'wb') as wav_writer:
    wav_writer.setnchannels(chunk.channels)
    wav_writer.setsampwidth(chunk.sample_width)
    wav_writer.setframerate(chunk.sample_rate)
    wav_writer.writeframes(chunk.pcm_bytes)
  return output.getvalue()


_FFMPEG_TIMEOUT_SECONDS = 60


def _run_ffmpeg(
  command: list[str],
  *,
  error_prefix: str,
  fallback_error: str,
) -> None:
  try:
    process = subprocess.run(
      command, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_SECONDS
    )
  except subprocess.TimeoutExpired as exc:
    raise RuntimeError(
      f'{error_prefix}: ffmpeg timed out after {_FFMPEG_TIMEOUT_SECONDS}s'
    ) from exc

  if process.returncode == 0:
    return

  stderr = process.stderr.strip() or fallback_error
  raise RuntimeError(f'{error_prefix}: {stderr}')


def merge_chunks(chunks: list[AudioChunk]) -> AudioChunk:
  if not chunks:
    raise ValueError('cannot merge empty audio chunk list')

  first = chunks[0]
  first_format = _chunk_format(first)
  pcm_parts = [first.pcm_bytes]
  for chunk in chunks[1:]:
    chunk_format = _chunk_format(chunk)
    if chunk_format != first_format:
      raise RuntimeError(
        'engine chunks returned incompatible audio formats '
        f'({_format_signature(first_format)} vs '
        f'{_format_signature(chunk_format)})'
      )
    pcm_parts.append(chunk.pcm_bytes)

  return AudioChunk(
    pcm_bytes=b''.join(pcm_parts),
    sample_rate=first.sample_rate,
    channels=first.channels,
    sample_width=first.sample_width,
  )


def _encode_mp3(wav_payload: bytes, ffmpeg_path: str) -> bytes:
  with TemporaryDirectory(prefix='tts-gateway-encode-') as temp_dir:
    wav_path = Path(temp_dir) / 'input.wav'
    mp3_path = Path(temp_dir) / 'output.mp3'
    wav_path.write_bytes(wav_payload)

    command = [
      ffmpeg_path,
      '-hide_banner',
      '-loglevel',
      'error',
      '-y',
      '-i',
      str(wav_path),
      '-codec:a',
      'libmp3lame',
      '-b:a',
      '128k',
      str(mp3_path),
    ]
    _run_ffmpeg(
      command,
      error_prefix='ffmpeg mp3 encode failed',
      fallback_error='unknown ffmpeg encode error',
    )
    return mp3_path.read_bytes()


def align_chunk_format(
  chunk: AudioChunk, reference: AudioChunk, ffmpeg_path: str
) -> AudioChunk:
  if _chunk_format(chunk) == _chunk_format(reference):
    return chunk

  sample_format = SAMPLE_FORMAT_BY_WIDTH.get(reference.sample_width)
  if sample_format is None:
    raise RuntimeError(
      f'unsupported sample width for alignment: {reference.sample_width}'
    )

  with TemporaryDirectory(prefix='tts-gateway-align-') as temp_dir:
    input_wav = Path(temp_dir) / 'input.wav'
    output_wav = Path(temp_dir) / 'aligned.wav'
    input_wav.write_bytes(chunk_to_wav_bytes(chunk))

    command = [
      ffmpeg_path,
      '-hide_banner',
      '-loglevel',
      'error',
      '-y',
      '-i',
      str(input_wav),
      '-ar',
      str(reference.sample_rate),
      '-ac',
      str(reference.channels),
      '-sample_fmt',
      sample_format,
      '-f',
      'wav',
      str(output_wav),
    ]
    _run_ffmpeg(
      command,
      error_prefix='ffmpeg alignment failed',
      fallback_error='unknown ffmpeg align error',
    )

    return wav_bytes_to_chunk(output_wav.read_bytes())


def encode_output(
  chunk: AudioChunk, output_format: OutputFormat, ffmpeg_path: str
) -> tuple[bytes, str]:
  wav_payload = chunk_to_wav_bytes(chunk)
  if output_format == 'wav':
    return wav_payload, 'audio/wav'

  if output_format == 'mp3':
    return _encode_mp3(wav_payload, ffmpeg_path), 'audio/mpeg'

  raise RuntimeError(f'unsupported output format: {output_format}')


def _chunk_format(chunk: AudioChunk) -> tuple[int, int, int]:
  return (chunk.sample_rate, chunk.channels, chunk.sample_width)


def _format_signature(chunk_format: tuple[int, int, int]) -> str:
  sample_rate, channels, sample_width = chunk_format
  return f'{sample_rate}/{channels}/{sample_width}'
