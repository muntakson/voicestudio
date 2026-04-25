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
            edited_transcript TEXT,
            rewritten_text TEXT,
            generated_audio_filename TEXT,
            generated_audio_size INTEGER DEFAULT 0,
            generated_audio_duration REAL DEFAULT 0,
            status TEXT DEFAULT 'created',
            asr_model TEXT,
            asr_elapsed REAL DEFAULT 0,
            asr_audio_duration REAL DEFAULT 0,
            asr_cost REAL DEFAULT 0,
            fix_typos_model TEXT,
            fix_typos_input_tokens INTEGER DEFAULT 0,
            fix_typos_output_tokens INTEGER DEFAULT 0,
            fix_typos_elapsed REAL DEFAULT 0,
            fix_typos_cost REAL DEFAULT 0,
            rewrite_model TEXT,
            rewrite_input_tokens INTEGER DEFAULT 0,
            rewrite_output_tokens INTEGER DEFAULT 0,
            rewrite_elapsed REAL DEFAULT 0,
            rewrite_cost REAL DEFAULT 0,
            tts_engine TEXT,
            tts_model TEXT,
            tts_text_chars INTEGER DEFAULT 0,
            tts_elapsed REAL DEFAULT 0,
            tts_cost REAL DEFAULT 0,
            total_cost REAL DEFAULT 0
        )
    """)
    _migrate(db)
    db.commit()


def _migrate(db: sqlite3.Connection):
    cursor = db.execute("PRAGMA table_info(projects)")
    existing = {row[1] for row in cursor.fetchall()}
    new_cols = [
        ("edited_transcript", "TEXT"),
        ("asr_model", "TEXT"),
        ("asr_elapsed", "REAL DEFAULT 0"),
        ("asr_audio_duration", "REAL DEFAULT 0"),
        ("asr_cost", "REAL DEFAULT 0"),
        ("fix_typos_model", "TEXT"),
        ("fix_typos_input_tokens", "INTEGER DEFAULT 0"),
        ("fix_typos_output_tokens", "INTEGER DEFAULT 0"),
        ("fix_typos_elapsed", "REAL DEFAULT 0"),
        ("fix_typos_cost", "REAL DEFAULT 0"),
        ("rewrite_model", "TEXT"),
        ("rewrite_input_tokens", "INTEGER DEFAULT 0"),
        ("rewrite_output_tokens", "INTEGER DEFAULT 0"),
        ("rewrite_elapsed", "REAL DEFAULT 0"),
        ("rewrite_cost", "REAL DEFAULT 0"),
        ("generated_audio_duration", "REAL DEFAULT 0"),
        ("tts_engine", "TEXT"),
        ("tts_model", "TEXT"),
        ("tts_text_chars", "INTEGER DEFAULT 0"),
        ("tts_elapsed", "REAL DEFAULT 0"),
        ("tts_cost", "REAL DEFAULT 0"),
        ("total_cost", "REAL DEFAULT 0"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            db.execute(f"ALTER TABLE projects ADD COLUMN {col_name} {col_def}")


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
    "num_speakers", "llm_model", "edited_transcript", "rewritten_text",
    "generated_audio_filename", "generated_audio_size", "generated_audio_duration",
    "status",
    "asr_model", "asr_elapsed", "asr_audio_duration", "asr_cost",
    "fix_typos_model", "fix_typos_input_tokens", "fix_typos_output_tokens", "fix_typos_elapsed", "fix_typos_cost",
    "rewrite_model", "rewrite_input_tokens", "rewrite_output_tokens", "rewrite_elapsed", "rewrite_cost",
    "tts_engine", "tts_model", "tts_text_chars", "tts_elapsed", "tts_cost",
    "total_cost",
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
