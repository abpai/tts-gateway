from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tts_gateway.engines.kokoro_native import KokoroNativeEngine


def _fake_torch(*, cuda_available: bool, mps_available: bool) -> SimpleNamespace:
  return SimpleNamespace(
    cuda=SimpleNamespace(is_available=lambda: cuda_available),
    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
  )


def test_resolve_device_cpu_mode() -> None:
  assert KokoroNativeEngine._resolve_device('cpu') == 'cpu'


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=True, mps_available=True)
  )
  assert KokoroNativeEngine._resolve_device('auto') == 'cuda'


def test_resolve_device_auto_uses_mps_when_cuda_missing(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=False, mps_available=True)
  )
  assert KokoroNativeEngine._resolve_device('auto') == 'mps'


def test_resolve_device_cuda_mode_falls_back_when_unavailable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=False, mps_available=True)
  )
  assert KokoroNativeEngine._resolve_device('cuda') == 'cpu'


def test_resolve_device_mps_mode_falls_back_when_unavailable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'torch', _fake_torch(cuda_available=True, mps_available=False)
  )
  assert KokoroNativeEngine._resolve_device('mps') == 'cpu'
