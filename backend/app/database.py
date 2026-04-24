"""SQLite3 database for biography project management."""

import json
import os
import sqlite3
import threading
from datetime import datetime

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_base_dir, "projects.db")
ARTIFACTS_DIR = os.path.join(_base_dir, "artifacts")
AUDIO_FILES_DIR = os.path.join(_base_dir, "audio_files")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)

_local = threading.local()


def get_db() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source_audio_filename TEXT,
            source_audio_original_name TEXT,
            source_audio_size INTEGER DEFAULT 0,
            transcript_json TEXT,
            transcript_text TEXT,
            num_speakers INTEGER DEFAULT 2,
            llm_model TEXT,
            rewritten_text TEXT,
            generated_audio_filename TEXT,
            generated_audio_size INTEGER DEFAULT 0,
            status TEXT DEFAULT 'created'
        )
    """)
    db.commit()


def list_projects() -> list[dict]:
    db = get_db()
    rows = db.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict | None:
    db = get_db()
    row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return dict(row) if row else None


def create_project(project_id: str, name: str, created_at: str) -> dict:
    db = get_db()
    db.execute(
        "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
        (project_id, name, created_at),
    )
    db.commit()
    return get_project(project_id)


_ALLOWED_FIELDS = {
    "name", "source_audio_filename", "source_audio_original_name",
    "source_audio_size", "transcript_json", "transcript_text",
    "num_speakers", "llm_model", "rewritten_text",
    "generated_audio_filename", "generated_audio_size", "status",
}


def update_project(project_id: str, **fields) -> dict | None:
    fields = {k: v for k, v in fields.items() if k in _ALLOWED_FIELDS}
    if not fields:
        return get_project(project_id)
    db = get_db()
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [project_id]
    db.execute(f"UPDATE projects SET {sets} WHERE id = ?", values)
    db.commit()
    return get_project(project_id)


def delete_project(project_id: str) -> bool:
    db = get_db()
    cursor = db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    return cursor.rowcount > 0
