"""Tests for text normalization and chunking."""

from __future__ import annotations

from tts_gateway.chunking import chunk_text, normalize_text, stream_chunk_text


def test_normalize_text_strips_markdown_syntax() -> None:
  text = """
  # Heading
  Visit [Anthropic](https://www.anthropic.com) for `docs`.
  - **Bold** item
  1. _Italic_ item
  Raw URL: https://example.com/test
  """

  assert normalize_text(text) == (
    'Heading Visit Anthropic for docs. Bold item Italic item Raw URL:'
  )


def test_stream_chunk_text_short_text_single_chunk() -> None:
  text = 'Hello world.'
  chunks = stream_chunk_text(text, first_chunk_max_chars=180, chunk_max_chars=500)
  assert chunks == ['Hello world.']


def test_stream_chunk_text_several_short_sentences() -> None:
  text = 'First. Second. Third. Fourth.'
  chunks = stream_chunk_text(text, first_chunk_max_chars=12, chunk_max_chars=30)
  assert chunks[0] == 'First.'
  assert len(chunks[0]) <= 12
  assert ' '.join(chunks) == normalize_text(text)


def test_stream_chunk_text_long_first_sentence() -> None:
  text = (
    'This is a very long opening sentence that keeps going without punctuation '
    'for quite a while before it finally ends. Then a short tail.'
  )
  chunks = stream_chunk_text(text, first_chunk_max_chars=40, chunk_max_chars=120)
  assert len(chunks[0]) <= 40
  assert chunks[0] == chunk_text(text, 40)[0]
  assert len(chunks) > 1
  assert all(len(chunk) <= 120 for chunk in chunks[1:])


def test_stream_chunk_text_markdown_noisy_text() -> None:
  text = """
  # Notes
  See [docs](https://example.com) and `inline` code.
  - **Bold** point
  """
  normalized = normalize_text(text)
  chunks = stream_chunk_text(text, first_chunk_max_chars=25, chunk_max_chars=80)
  assert chunks[0] == chunk_text(normalized, 25)[0]
  assert ' '.join(chunks) == normalized


def test_stream_chunk_text_no_punctuation_prose() -> None:
  text = (
    'one two three four five six seven eight nine ten eleven twelve '
    'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'
  )
  chunks = stream_chunk_text(text, first_chunk_max_chars=30, chunk_max_chars=60)
  assert len(chunks[0]) <= 30
  assert len(chunks) > 1
  assert all(len(chunk) <= 60 for chunk in chunks[1:])


def test_stream_chunk_text_matches_disk_plan_when_first_limit_covers_all() -> None:
  text = 'Short selection.'
  disk_chunks = chunk_text(text, 500)
  stream_chunks = stream_chunk_text(
    text, first_chunk_max_chars=180, chunk_max_chars=500
  )
  assert stream_chunks == disk_chunks
