from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
