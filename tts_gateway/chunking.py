from __future__ import annotations

import re

SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')
WHITESPACE = re.compile(r'\s+')


def normalize_text(text: str) -> str:
  return WHITESPACE.sub(' ', text).strip()


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
  words = segment.split(' ')
  chunks: list[str] = []
  current: list[str] = []
  current_len = 0

  for word in words:
    if not word:
      continue

    projected = current_len + (1 if current else 0) + len(word)
    if projected <= max_chars:
      current.append(word)
      current_len = projected
      continue

    if current:
      chunks.append(' '.join(current).strip())
      current = []
      current_len = 0

    if len(word) <= max_chars:
      current.append(word)
      current_len = len(word)
      continue

    # A single token can still exceed max_chars (for example very long URLs).
    start = 0
    while start < len(word):
      end = min(start + max_chars, len(word))
      chunks.append(word[start:end])
      start = end

  if current:
    chunks.append(' '.join(current).strip())

  return [chunk for chunk in chunks if chunk]


def chunk_text(text: str, max_chars: int) -> list[str]:
  normalized = normalize_text(text)
  if not normalized:
    return []

  if len(normalized) <= max_chars:
    return [normalized]

  sentences = SENTENCE_BOUNDARY.split(normalized)
  segments: list[str] = []
  for sentence in sentences:
    cleaned = sentence.strip()
    if not cleaned:
      continue
    if len(cleaned) <= max_chars:
      segments.append(cleaned)
      continue
    segments.extend(_split_long_segment(cleaned, max_chars))

  chunks: list[str] = []
  current = ''

  for segment in segments:
    if not current:
      current = segment
      continue

    projected = f'{current} {segment}'
    if len(projected) <= max_chars:
      current = projected
      continue

    chunks.append(current.strip())
    current = segment

  if current:
    chunks.append(current.strip())

  return chunks
