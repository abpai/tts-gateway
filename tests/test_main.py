from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import _make_config
from tts_gateway.main import create_app


def test_health_reports_chunk_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr('tts_gateway.gateway.os.cpu_count', lambda: 8)
  app = create_app(_make_config(kokoro_enabled=False, chunk_max_chars=3000))
  client = TestClient(app)

  response = client.get('/health')

  assert response.status_code == 200
  assert response.json()['chunkConcurrency'] == 4
