"""Tests for config parsing, focusing on new native-engine fields."""

from __future__ import annotations

import pytest

from tts_gateway.config import (
  _parse_bool,
  _parse_device_mode,
  load_config,
)

# ---------------------------------------------------------------------------
# _parse_bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  'env_val,expected',
  [
    ('true', True),
    ('True', True),
    ('TRUE', True),
    ('1', True),
    ('yes', True),
    ('on', True),
    ('false', False),
    ('False', False),
    ('0', False),
    ('no', False),
    ('off', False),
  ],
)
def test_parse_bool_valid(
  monkeypatch: pytest.MonkeyPatch, env_val: str, expected: bool
) -> None:
  monkeypatch.setenv('TEST_FLAG', env_val)
  assert _parse_bool('TEST_FLAG', False) == expected


def test_parse_bool_default(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv('TEST_FLAG', raising=False)
  assert _parse_bool('TEST_FLAG', True) is True
  assert _parse_bool('TEST_FLAG', False) is False


def test_parse_bool_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('TEST_FLAG', 'maybe')
  with pytest.raises(ValueError, match='boolean'):
    _parse_bool('TEST_FLAG', False)


# ---------------------------------------------------------------------------
# _parse_device_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', ['auto', 'cuda', 'mps', 'cpu'])
def test_parse_device_mode_valid(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
  monkeypatch.setenv('TEST_DEV', mode)
  assert _parse_device_mode('TEST_DEV', 'auto') == mode


def test_parse_device_mode_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('TEST_DEV', 'tpu')
  with pytest.raises(ValueError, match='auto, cuda, mps, cpu'):
    _parse_device_mode('TEST_DEV', 'auto')


# ---------------------------------------------------------------------------
# load_config — mode resolution
# ---------------------------------------------------------------------------


def _set_minimal_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
  """Set the minimal env vars for load_config to succeed."""
  defaults = {
    'TTS_PRIMARY_ENGINE': 'kokoro',
    'TTS_FALLBACK_ENGINE': 'none',
    'TTS_OUTPUT_FORMAT': 'wav',
    'KOKORO_TTS_ENABLED': 'true',
    'POCKET_TTS_ENABLED': 'false',
    'TTS_DEVICE_MODE': 'auto',
    'TTS_MODELS_DIR': '/tmp/models',
  }
  defaults.update(overrides)
  for key, val in defaults.items():
    monkeypatch.setenv(key, val)


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
  _set_minimal_env(monkeypatch)
  cfg = load_config()
  assert cfg.kokoro_enabled is True
  assert cfg.pocket_enabled is False
  assert cfg.device_mode == 'auto'
  assert cfg.output_format == 'wav'
  assert cfg.chunk_max_chars == 3000
  assert cfg.request_timeout_seconds == 3600
  assert cfg.fallback_engine is None
  assert cfg.default_voice is None
  assert cfg.bind_host == '127.0.0.1'
  assert cfg.bind_port == 8000


def test_load_config_both_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
  _set_minimal_env(
    monkeypatch,
    KOKORO_TTS_ENABLED='false',
    POCKET_TTS_ENABLED='false',
  )
  cfg = load_config()
  assert cfg.kokoro_enabled is False
  assert cfg.pocket_enabled is False


def test_load_config_default_voice(monkeypatch: pytest.MonkeyPatch) -> None:
  _set_minimal_env(monkeypatch, TTS_DEFAULT_VOICE='af_bella')
  cfg = load_config()
  assert cfg.default_voice == 'af_bella'
