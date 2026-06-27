"""Tests for scripts/cosyvoice_local_sidecar.py."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  'cosyvoice_local_sidecar',
  ROOT / 'scripts' / 'cosyvoice_local_sidecar.py',
)
assert SPEC is not None and SPEC.loader is not None
sidecar = importlib.util.module_from_spec(SPEC)
sys.modules['cosyvoice_local_sidecar'] = sidecar
SPEC.loader.exec_module(sidecar)


def _settings(**overrides: object) -> sidecar.LocalSidecarSettings:
  defaults: dict[str, object] = {
    'mode': 'sft',
    'cosyvoice_repo': ROOT,
    'model_dir': ROOT,
    'default_voice': 'default-spk',
    'prompt_text': None,
    'prompt_wav_path': None,
    'instruct_text': None,
    'sample_rate': 22050,
    'channels': 1,
    'pcm_format': 's16le',
    'sample_width': 2,
  }
  defaults.update(overrides)
  return sidecar.LocalSidecarSettings(**defaults)


class _FakeTensor:
  def __init__(self, values: np.ndarray) -> None:
    self._values = values

  def detach(self) -> _FakeTensor:
    return self

  def cpu(self) -> _FakeTensor:
    return self

  def numpy(self) -> np.ndarray:
    return self._values


class _FakeModel:
  def __init__(self) -> None:
    self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

  def _record(
    self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
  ) -> list[dict[str, np.ndarray]]:
    self.calls.append((name, args, kwargs))
    return [{'tts_speech': np.array([0.0, 0.5], dtype=np.float32)}]

  def inference_sft(
    self, text: str, spk_id: str, *, stream: bool = False
  ) -> list[dict[str, np.ndarray]]:
    return self._record('inference_sft', (text, spk_id), {'stream': stream})

  def inference_zero_shot(
    self,
    text: str,
    prompt_text: str,
    prompt_wav: str,
    *,
    stream: bool = False,
  ) -> list[dict[str, np.ndarray]]:
    return self._record(
      'inference_zero_shot',
      (text, prompt_text, prompt_wav),
      {'stream': stream},
    )

  def inference_cross_lingual(
    self, text: str, prompt_wav: str, *, stream: bool = False
  ) -> list[dict[str, np.ndarray]]:
    return self._record(
      'inference_cross_lingual',
      (text, prompt_wav),
      {'stream': stream},
    )

  def inference_instruct(
    self,
    text: str,
    spk_id: str,
    instruct_text: str,
    *,
    stream: bool = False,
  ) -> list[dict[str, np.ndarray]]:
    return self._record(
      'inference_instruct',
      (text, spk_id, instruct_text),
      {'stream': stream},
    )

  def inference_instruct2(
    self,
    text: str,
    instruct_text: str,
    prompt_wav: str,
    *,
    stream: bool = False,
  ) -> list[dict[str, np.ndarray]]:
    return self._record(
      'inference_instruct2',
      (text, instruct_text, prompt_wav),
      {'stream': stream},
    )


@pytest.fixture
def cosyvoice_repo(tmp_path: Path) -> Path:
  repo = tmp_path / 'CosyVoice'
  repo.mkdir()
  matcha = repo / 'third_party' / 'Matcha-TTS'
  matcha.mkdir(parents=True)
  return repo


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
  path = tmp_path / 'model'
  path.mkdir()
  return path


@pytest.fixture
def prompt_wav(tmp_path: Path) -> Path:
  path = tmp_path / 'prompt.wav'
  path.write_bytes(b'RIFFfake-wav')
  return path


def test_parse_args_requires_repo_and_model(
  cosyvoice_repo: Path,
  model_dir: Path,
) -> None:
  args = sidecar.parse_args(
    [
      '--cosyvoice-repo',
      str(cosyvoice_repo),
      '--model-dir',
      str(model_dir),
      '--default-voice',
      'spk-1',
    ]
  )
  assert args.mode == 'sft'
  assert args.host == '127.0.0.1'
  assert args.port == 50000
  assert args.sample_rate == 22050
  assert args.debug is False


def test_settings_sft_requires_default_voice(
  cosyvoice_repo: Path,
  model_dir: Path,
) -> None:
  args = sidecar.parse_args(
    [
      '--cosyvoice-repo',
      str(cosyvoice_repo),
      '--model-dir',
      str(model_dir),
    ]
  )
  with pytest.raises(SystemExit, match='default-voice'):
    sidecar.settings_from_args(args)


@pytest.mark.parametrize(
  ('mode', 'extra_argv', 'match'),
  [
    ('zero-shot', ['--prompt-text', 'hello'], 'prompt-wav'),
    ('cross-lingual', [], 'prompt-wav'),
    ('instruct', ['--default-voice', 'spk-1'], 'instruct-text'),
    ('instruct2', ['--instruct-text', 'slow'], 'prompt-wav'),
  ],
)
def test_settings_rejects_missing_mode_requirements(
  cosyvoice_repo: Path,
  model_dir: Path,
  mode: str,
  extra_argv: list[str],
  match: str,
) -> None:
  argv = [
    '--cosyvoice-repo',
    str(cosyvoice_repo),
    '--model-dir',
    str(model_dir),
    '--mode',
    mode,
    *extra_argv,
  ]
  with pytest.raises(SystemExit, match=match):
    sidecar.settings_from_args(sidecar.parse_args(argv))


def test_english_narration_sets_default_instruct_text(
  cosyvoice_repo: Path,
  model_dir: Path,
) -> None:
  args = sidecar.parse_args(
    [
      '--cosyvoice-repo',
      str(cosyvoice_repo),
      '--model-dir',
      str(model_dir),
      '--mode',
      'instruct',
      '--default-voice',
      'spk-1',
      '--english-narration',
    ]
  )
  settings = sidecar.settings_from_args(args)
  assert settings.instruct_text == sidecar.DEFAULT_ENGLISH_NARRATION_INSTRUCT


def test_english_narration_still_requires_zero_shot_prompt_text(
  cosyvoice_repo: Path,
  model_dir: Path,
  prompt_wav: Path,
) -> None:
  args = sidecar.parse_args(
    [
      '--cosyvoice-repo',
      str(cosyvoice_repo),
      '--model-dir',
      str(model_dir),
      '--mode',
      'zero-shot',
      '--english-narration',
      '--prompt-wav',
      str(prompt_wav),
    ]
  )
  with pytest.raises(SystemExit, match='--prompt-text must not be empty'):
    sidecar.settings_from_args(args)


def test_english_narration_preserves_zero_shot_prompt_transcript(
  cosyvoice_repo: Path,
  model_dir: Path,
  prompt_wav: Path,
) -> None:
  args = sidecar.parse_args(
    [
      '--cosyvoice-repo',
      str(cosyvoice_repo),
      '--model-dir',
      str(model_dir),
      '--mode',
      'zero-shot',
      '--english-narration',
      '--prompt-text',
      'This is the transcript of the English reference.',
      '--prompt-wav',
      str(prompt_wav),
    ]
  )
  settings = sidecar.settings_from_args(args)
  assert settings.prompt_text == 'This is the transcript of the English reference.'


def test_settings_rejects_invalid_repo(
  model_dir: Path,
) -> None:
  args = sidecar.parse_args(
    [
      '--cosyvoice-repo',
      '/tmp/does-not-exist-cosyvoice-repo',
      '--model-dir',
      str(model_dir),
      '--default-voice',
      'spk-1',
    ]
  )
  with pytest.raises(SystemExit, match='cosyvoice-repo'):
    sidecar.settings_from_args(args)


def test_settings_rejects_missing_prompt_wav_file(
  cosyvoice_repo: Path,
  model_dir: Path,
) -> None:
  args = sidecar.parse_args(
    [
      '--cosyvoice-repo',
      str(cosyvoice_repo),
      '--model-dir',
      str(model_dir),
      '--mode',
      'cross-lingual',
      '--prompt-wav',
      '/tmp/does-not-exist-prompt.wav',
    ]
  )
  with pytest.raises(SystemExit, match='prompt-wav'):
    sidecar.settings_from_args(args)


def test_configure_cosyvoice_import_path_inserts_repo_entries(
  cosyvoice_repo: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  original_path = sys.path.copy()
  monkeypatch.setattr(sys, 'path', original_path.copy())
  sidecar.configure_cosyvoice_import_path(cosyvoice_repo)
  matcha = str((cosyvoice_repo / 'third_party' / 'Matcha-TTS').resolve())
  repo = str(cosyvoice_repo.resolve())
  assert sys.path[0] == repo
  assert sys.path[1] == matcha


def test_load_automodel_uses_configured_import_path(
  cosyvoice_repo: Path,
  model_dir: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  inserted: list[Path] = []

  def _configure(repo: Path) -> None:
    inserted.append(repo)

  class _AutoModel:
    last_model_dir: str | None = None

    def __init__(self, *, model_dir: str) -> None:
      _AutoModel.last_model_dir = model_dir

  fake_module = type(sys)('cosyvoice.cli.cosyvoice')
  fake_module.AutoModel = _AutoModel  # type: ignore[attr-defined]
  monkeypatch.setattr(sidecar, 'configure_cosyvoice_import_path', _configure)
  monkeypatch.setitem(sys.modules, 'cosyvoice.cli.cosyvoice', fake_module)

  settings = _settings(cosyvoice_repo=cosyvoice_repo, model_dir=model_dir)
  sidecar.load_automodel(settings)

  assert inserted == [cosyvoice_repo]
  assert _AutoModel.last_model_dir == str(model_dir)


def test_speech_to_pcm_bytes_clamps_and_casts() -> None:
  pcm = sidecar.speech_to_pcm_bytes(np.array([2.0, -2.0, 0.5, -0.5]))
  samples = struct.unpack('<4h', pcm)
  assert samples == (32767, -32767, 16383, -16383)


def test_speech_to_pcm_bytes_supports_tensor_like_values() -> None:
  pcm = sidecar.speech_to_pcm_bytes(_FakeTensor(np.array([0.0, 1.0])))
  samples = struct.unpack('<2h', pcm)
  assert samples == (0, 32767)


@pytest.mark.parametrize(
  ('mode', 'settings_overrides', 'voice', 'expected_method', 'expected_args'),
  [
    (
      'sft',
      {},
      'custom-spk',
      'inference_sft',
      ('hello', 'custom-spk'),
    ),
    (
      'zero-shot',
      {
        'mode': 'zero-shot',
        'default_voice': None,
        'prompt_text': 'prompt transcript',
        'prompt_wav_path': Path('/tmp/prompt.wav'),
      },
      None,
      'inference_zero_shot',
      ('hello', 'prompt transcript', '/tmp/prompt.wav'),
    ),
    (
      'cross-lingual',
      {
        'mode': 'cross-lingual',
        'default_voice': None,
        'prompt_wav_path': Path('/tmp/prompt.wav'),
      },
      None,
      'inference_cross_lingual',
      ('hello', '/tmp/prompt.wav'),
    ),
    (
      'instruct',
      {
        'mode': 'instruct',
        'instruct_text': 'speak slowly',
      },
      'custom-spk',
      'inference_instruct',
      ('hello', 'custom-spk', 'speak slowly'),
    ),
    (
      'instruct2',
      {
        'mode': 'instruct2',
        'default_voice': None,
        'instruct_text': 'speak slowly',
        'prompt_wav_path': Path('/tmp/prompt.wav'),
      },
      None,
      'inference_instruct2',
      ('hello', 'speak slowly', '/tmp/prompt.wav'),
    ),
  ],
)
def test_dispatch_inference_calls_expected_model_method(
  mode: str,
  settings_overrides: dict[str, object],
  voice: str | None,
  expected_method: str,
  expected_args: tuple[str, ...],
) -> None:
  settings = _settings(**settings_overrides)
  model = _FakeModel()
  chunks = list(sidecar.iter_model_pcm(settings, model, text='hello', voice=voice))

  assert len(chunks) == 1
  assert model.calls[0][0] == expected_method
  assert model.calls[0][1] == expected_args
  assert model.calls[0][2]['stream'] is True


def test_health_response_shape(prompt_wav: Path) -> None:
  app = sidecar.create_app(
    _settings(
      mode='zero-shot',
      default_voice=None,
      prompt_text='reference transcript',
      prompt_wav_path=prompt_wav,
    ),
    model_loader=lambda _settings: _FakeModel(),
  )
  with TestClient(app) as client:
    body = client.get('/health').json()

  assert body == {
    'status': 'ok',
    'backend': 'cosyvoice-local',
    'mode': 'zero-shot',
    'modelDir': str(ROOT),
    'sampleRate': 22050,
    'defaultVoiceConfigured': False,
    'promptTextConfigured': True,
    'promptWavConfigured': True,
    'instructTextConfigured': False,
    'promptWavBasename': 'prompt.wav',
  }
  assert '/Users/' not in str(body.get('promptWavBasename', ''))
  assert str(prompt_wav) not in body.values()


def test_stream_returns_pcm_headers_and_bytes() -> None:
  app = sidecar.create_app(
    _settings(),
    model_loader=lambda _settings: _FakeModel(),
  )
  with TestClient(app) as client:
    response = client.post('/v1/tts/stream', json={'text': 'hello world'})

  assert response.status_code == 200
  assert response.headers['content-type'] == 'audio/raw'
  assert response.headers['x-tts-sample-rate'] == '22050'
  assert response.headers['x-tts-channels'] == '1'
  assert response.headers['x-tts-pcm-format'] == 's16le'
  assert response.headers['x-tts-sample-width'] == '2'
  assert response.headers['x-tts-backend'] == 'cosyvoice-local'
  assert len(response.content) == 4


def test_stream_blank_text_returns_422() -> None:
  app = sidecar.create_app(
    _settings(),
    model_loader=lambda _settings: _FakeModel(),
  )
  with TestClient(app) as client:
    response = client.post('/v1/tts/stream', json={'text': '   '})

  assert response.status_code == 422


def test_stream_blank_voice_uses_default() -> None:
  model = _FakeModel()
  app = sidecar.create_app(_settings(), model_loader=lambda _settings: model)
  with TestClient(app) as client:
    client.post('/v1/tts/stream', json={'text': 'hello', 'voice': '   '})

  assert model.calls[0][1][1] == 'default-spk'
