"""Tests for text normalization and chunking."""

from __future__ import annotations

from tts_gateway.chunking import normalize_text


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
