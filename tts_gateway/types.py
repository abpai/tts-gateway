"""Core value objects for the TTS gateway.

Four types, no behavior beyond serialization:
  SynthesisSpec  -- immutable user intent
  RenderPlan     -- deterministic chunked execution plan
  ArtifactRef    -- pointer to completed output on disk
  JobView        -- public-facing job status
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tts_gateway.config import OutputFormat

JobStatus = Literal['queued', 'running', 'encoding', 'ready', 'failed']


@dataclass(frozen=True)
class SynthesisSpec:
  """Immutable description of what to synthesize."""

  text: str
  voice: str
  output_format: OutputFormat
  chunk_max_chars: int = 500
  pipeline_version: str = '1'

  def to_json(self) -> str:
    return json.dumps(
      {
        'text': self.text,
        'voice': self.voice,
        'output_format': self.output_format,
        'chunk_max_chars': self.chunk_max_chars,
        'pipeline_version': self.pipeline_version,
      },
      sort_keys=True,
      separators=(',', ':'),
    )

  @classmethod
  def from_json(cls, raw: str) -> SynthesisSpec:
    data = json.loads(raw)
    return cls(
      text=data['text'],
      voice=data.get('voice', ''),
      output_format=data.get('output_format', 'wav'),
      chunk_max_chars=data.get('chunk_max_chars', 500),
      pipeline_version=data.get('pipeline_version', '1'),
    )

  @functools.cached_property
  def content_hash(self) -> str:
    return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True)
class RenderPlan:
  """Text split into ordered chunks, ready for synthesis."""

  request_hash: str
  chunks: tuple[str, ...]
  voice: str
  output_format: OutputFormat


@dataclass(frozen=True)
class ArtifactRef:
  """Pointer to a completed synthesis on disk."""

  request_hash: str
  output_path: Path
  content_type: str
  chunks_total: int
  duration_ms: int


@dataclass(frozen=True)
class JobView:
  """Public-facing job status."""

  key: str
  status: JobStatus
  created_at: str
  started_at: str | None
  completed_at: str | None
  chunks_total: int | None
  chunks_done: int
  content_type: str | None
  error: str | None
