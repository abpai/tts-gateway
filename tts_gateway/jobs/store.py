"""SQLite-backed job store for content-addressed TTS synthesis jobs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

JobStatus = Literal['queued', 'running', 'encoding', 'ready', 'failed']

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
  key            TEXT PRIMARY KEY,
  status         TEXT NOT NULL DEFAULT 'queued',
  request_json   TEXT NOT NULL,
  chunks_total   INTEGER,
  chunks_done    INTEGER NOT NULL DEFAULT 0,
  content_type   TEXT,
  artifact_path  TEXT,
  error          TEXT,
  created_at     TEXT NOT NULL,
  started_at     TEXT,
  completed_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


@dataclass
class JobRecord:
  key: str
  status: JobStatus
  request_json: str
  chunks_total: int | None
  chunks_done: int
  content_type: str | None
  artifact_path: str | None
  error: str | None
  created_at: str
  started_at: str | None
  completed_at: str | None


def _now_iso() -> str:
  return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _row_to_record(row: sqlite3.Row) -> JobRecord:
  return JobRecord(
    key=row['key'],
    status=row['status'],
    request_json=row['request_json'],
    chunks_total=row['chunks_total'],
    chunks_done=row['chunks_done'],
    content_type=row['content_type'],
    artifact_path=row['artifact_path'],
    error=row['error'],
    created_at=row['created_at'],
    started_at=row['started_at'],
    completed_at=row['completed_at'],
  )


class JobStore:
  """Thin SQLite wrapper for job state management.

  Thread-safe: each method opens its own cursor.
  WAL mode enables concurrent reads during writes.
  """

  def __init__(self, db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    self._conn.execute('PRAGMA journal_mode=WAL')
    self._conn.execute('PRAGMA busy_timeout=5000')
    self._conn.executescript(_SCHEMA)

  def close(self) -> None:
    self._conn.close()

  def create_or_get(self, key: str, request_json: str) -> JobRecord:
    """Insert a new job or return the existing one.

    If a previous job with this key failed, requeue it so
    transient failures can be retried by resubmitting.
    """
    now = _now_iso()
    self._conn.execute(
      'INSERT OR IGNORE INTO jobs (key, status, request_json, created_at) VALUES (?, ?, ?, ?)',
      (key, 'queued', request_json, now),
    )
    # Reset failed/stale running jobs back to queued on resubmission
    self._conn.execute(
      """
      UPDATE jobs
      SET status = 'queued', error = NULL, started_at = NULL, completed_at = NULL
      WHERE key = ? AND status IN ('failed', 'running')
      """,
      (key,),
    )
    self._conn.commit()
    return self.get(key)

  def get(self, key: str) -> JobRecord | None:
    """Fetch a job by key."""
    row = self._conn.execute('SELECT * FROM jobs WHERE key = ?', (key,)).fetchone()
    if row is None:
      return None
    return _row_to_record(row)

  def claim_next(self) -> JobRecord | None:
    """Atomically claim the oldest queued job. Returns None if queue is empty."""
    now = _now_iso()
    cursor = self._conn.execute(
      """
      UPDATE jobs
      SET status = 'running', started_at = ?
      WHERE key = (
        SELECT key FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1
      )
      RETURNING *
      """,
      (now,),
    )
    row = cursor.fetchone()
    self._conn.commit()
    if row is None:
      return None
    return _row_to_record(row)

  def update_progress(self, key: str, chunks_done: int, chunks_total: int) -> None:
    """Update chunk progress for a running job."""
    self._conn.execute(
      'UPDATE jobs SET chunks_done = ?, chunks_total = ? WHERE key = ?',
      (chunks_done, chunks_total, key),
    )
    self._conn.commit()

  def mark_encoding(self, key: str) -> None:
    self._conn.execute("UPDATE jobs SET status = 'encoding' WHERE key = ?", (key,))
    self._conn.commit()

  def mark_ready(
    self,
    key: str,
    *,
    artifact_path: str,
    content_type: str,
    chunks_total: int,
  ) -> None:
    now = _now_iso()
    self._conn.execute(
      """
      UPDATE jobs
      SET status = 'ready', artifact_path = ?, content_type = ?,
          chunks_total = ?, chunks_done = ?, completed_at = ?
      WHERE key = ?
      """,
      (artifact_path, content_type, chunks_total, chunks_total, now, key),
    )
    self._conn.commit()

  def mark_failed(self, key: str, error: str) -> None:
    now = _now_iso()
    self._conn.execute(
      "UPDATE jobs SET status = 'failed', error = ?, completed_at = ? WHERE key = ?",
      (error, now, key),
    )
    self._conn.commit()

  def list_jobs(
    self, *, status: JobStatus | None = None, limit: int = 50
  ) -> list[JobRecord]:
    if status:
      rows = self._conn.execute(
        'SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?',
        (status, limit),
      ).fetchall()
    else:
      rows = self._conn.execute(
        'SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,)
      ).fetchall()
    return [_row_to_record(row) for row in rows]

  def reset_stale(self, older_than_seconds: int = 300) -> int:
    """Reset running jobs older than threshold back to queued. Returns count."""
    cursor = self._conn.execute(
      """
      UPDATE jobs
      SET status = 'queued', started_at = NULL
      WHERE status = 'running'
        AND started_at < datetime('now', ? || ' seconds')
      """,
      (f'-{older_than_seconds}',),
    )
    self._conn.commit()
    return cursor.rowcount
