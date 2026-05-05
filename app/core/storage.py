from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.schemas import utc_now


class SessionStore:
    def __init__(self, db_path: Path = settings.sqlite_path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    dataset_path TEXT,
                    user_query TEXT,
                    dataset_metadata TEXT,
                    report TEXT,
                    evaluation TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def save_state(self, state: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, created_at, updated_at, dataset_path, user_query,
                    dataset_metadata, report, evaluation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    dataset_path=excluded.dataset_path,
                    user_query=excluded.user_query,
                    dataset_metadata=excluded.dataset_metadata,
                    report=excluded.report,
                    evaluation=excluded.evaluation
                """,
                (
                    state["session_id"],
                    now,
                    now,
                    state.get("dataset_path"),
                    state.get("user_query"),
                    json.dumps(state.get("dataset_metadata", {})),
                    json.dumps(state.get("report", {})),
                    json.dumps(state.get("verification", {})),
                ),
            )
            for trace in state.get("traces", []):
                conn.execute(
                    "INSERT INTO traces (session_id, timestamp, event, payload) VALUES (?, ?, ?, ?)",
                    (
                        state["session_id"],
                        trace.get("timestamp", now),
                        trace.get("event", "trace"),
                        json.dumps(trace),
                    ),
                )

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, updated_at, dataset_path, user_query, evaluation FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "session_id": row[0],
                "updated_at": row[1],
                "dataset_path": row[2],
                "user_query": row[3],
                "evaluation": json.loads(row[4] or "{}"),
            }
            for row in rows
        ]

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            traces = conn.execute(
                "SELECT payload FROM traces WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        if not row:
            return None
        return {
            "session_id": row[0],
            "created_at": row[1],
            "updated_at": row[2],
            "dataset_path": row[3],
            "user_query": row[4],
            "dataset_metadata": json.loads(row[5] or "{}"),
            "report": json.loads(row[6] or "{}"),
            "evaluation": json.loads(row[7] or "{}"),
            "traces": [json.loads(item[0]) for item in traces],
        }


session_store = SessionStore()

