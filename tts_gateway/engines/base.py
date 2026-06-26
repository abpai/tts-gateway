from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard, runtime_checkable

import numpy as np


class EngineError(RuntimeError):
  pass


@dataclass(frozen=True)
class AudioChunk:
  pcm_bytes: bytes
  sample_rate: int
  channels: int
  sample_width: int

  @classmethod
  def from_float32(cls, audio: Any, *, sample_rate: int) -> AudioChunk:
    """Convert float32 audio array to 16-bit PCM AudioChunk."""
    pcm_int16 = (np.asarray(audio) * 32767).clip(-32768, 32767).astype(np.int16)
    return cls(
      pcm_bytes=pcm_int16.tobytes(),
      sample_rate=sample_rate,
      channels=1,
      sample_width=2,
    )


class TtsEngine(ABC):
  name: str

  @abstractmethod
  async def synthesize(self, text: str, *, voice: str | None = None) -> AudioChunk:
    """Generate a PCM chunk from text."""


@runtime_checkable
class StreamingTtsEngine(Protocol):
  """Engine that yields incremental PCM chunks for a full text input."""

  name: str

  def stream_synthesize(
    self, text: str, *, voice: str | None = None
  ) -> AsyncGenerator[AudioChunk, None]:
    """Yield ordered PCM chunks for text without gateway-side pre-chunking."""


def supports_streaming(engine: TtsEngine) -> TypeGuard[StreamingTtsEngine]:
  """Return True when engine exposes native streaming synthesis."""
  return isinstance(engine, StreamingTtsEngine)


# Alias for the redesign. Both names are exported; old code keeps working.
Engine = TtsEngine
