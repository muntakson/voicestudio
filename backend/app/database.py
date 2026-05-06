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
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS project_audio_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            original_name TEXT,
            file_size INTEGER DEFAULT 0,
            file_type TEXT DEFAULT 'source',
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_paf_project ON project_audio_files(project_id)
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS project_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            artifact_type TEXT NOT NULL DEFAULT 'text',
            file_size INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_pa_project ON project_artifacts(project_id)
    """)
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
            tts_text TEXT,
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


def _migrate_paf(db: sqlite3.Connection):
    cursor = db.execute("PRAGMA table_info(project_audio_files)")
    existing = {row[1] for row in cursor.fetchall()}
    if "file_type" not in existing:
        db.execute("ALTER TABLE project_audio_files ADD COLUMN file_type TEXT DEFAULT 'source'")


def _migrate(db: sqlite3.Connection):
    _migrate_paf(db)
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
        ("tts_text", "TEXT"),
        ("tts_engine", "TEXT"),
        ("tts_model", "TEXT"),
        ("tts_text_chars", "INTEGER DEFAULT 0"),
        ("tts_elapsed", "REAL DEFAULT 0"),
        ("tts_cost", "REAL DEFAULT 0"),
        ("total_cost", "REAL DEFAULT 0"),
        ("owner", "TEXT DEFAULT 'admin'"),
        ("poem_text", "TEXT"),
        ("poem_audio_filename", "TEXT"),
        ("poem_audio_duration", "REAL DEFAULT 0"),
        ("poem_image_prompt", "TEXT"),
        ("poem_image_filename", "TEXT"),
        ("poem_video_prompt", "TEXT"),
        ("poem_video_filename", "TEXT"),
        ("poem_gen_elapsed", "REAL DEFAULT 0"),
        ("poem_gen_summary", "TEXT"),
        ("category_id", "INTEGER"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            db.execute(f"ALTER TABLE projects ADD COLUMN {col_name} {col_def}")


def list_projects() -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT p.*, c.name AS category_name FROM projects p LEFT JOIN categories c ON p.category_id = c.id ORDER BY p.created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict | None:
    db = get_db()
    row = db.execute(
        "SELECT p.*, c.name AS category_name FROM projects p LEFT JOIN categories c ON p.category_id = c.id WHERE p.id = ?",
        (project_id,),
    ).fetchone()
    return dict(row) if row else None


def create_project(project_id: str, name: str, created_at: str, owner: str = "admin") -> dict:
    db = get_db()
    db.execute(
        "INSERT INTO projects (id, name, created_at, owner) VALUES (?, ?, ?, ?)",
        (project_id, name, created_at, owner),
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
    "tts_text", "tts_engine", "tts_model", "tts_text_chars", "tts_elapsed", "tts_cost",
    "total_cost", "owner",
    "poem_text", "poem_audio_filename", "poem_audio_duration",
    "poem_image_prompt", "poem_image_filename",
    "poem_video_prompt", "poem_video_filename", "poem_gen_elapsed",
    "poem_gen_summary",
    "category_id",
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
    db.execute("DELETE FROM project_audio_files WHERE project_id = ?", (project_id,))
    db.execute("DELETE FROM project_artifacts WHERE project_id = ?", (project_id,))
    cursor = db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    return cursor.rowcount > 0


def add_project_audio(project_id: str, filename: str, original_name: str = "", file_size: int = 0, file_type: str = "source"):
    db = get_db()
    existing = db.execute(
        "SELECT id FROM project_audio_files WHERE project_id = ? AND filename = ?",
        (project_id, filename),
    ).fetchone()
    if existing:
        return
    db.execute(
        "INSERT INTO project_audio_files (project_id, filename, original_name, file_size, file_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, filename, original_name or filename, file_size, file_type, datetime.utcnow().isoformat()),
    )
    db.commit()


def list_project_audio(project_id: str) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT filename, original_name, file_size, file_type, created_at FROM project_audio_files WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def upsert_project_artifact(project_id: str, filename: str, label: str, file_size: int = 0, artifact_type: str = "text"):
    db = get_db()
    existing = db.execute(
        "SELECT id FROM project_artifacts WHERE project_id = ? AND filename = ?",
        (project_id, filename),
    ).fetchone()
    now = datetime.utcnow().isoformat()
    if existing:
        db.execute(
            "UPDATE project_artifacts SET label = ?, file_size = ?, created_at = ? WHERE id = ?",
            (label, file_size, now, existing["id"]),
        )
    else:
        db.execute(
            "INSERT INTO project_artifacts (project_id, filename, label, artifact_type, file_size, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, filename, label, artifact_type, file_size, now),
        )
    db.commit()


def list_project_artifacts(project_id: str) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT filename, label, artifact_type, file_size, created_at FROM project_artifacts WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_project_artifacts(project_id: str):
    db = get_db()
    db.execute("DELETE FROM project_artifacts WHERE project_id = ?", (project_id,))
    db.commit()


# ---------------------------------------------------------------------------
# Category CRUD
# ---------------------------------------------------------------------------

def list_categories() -> list[dict]:
    db = get_db()
    rows = db.execute("SELECT * FROM categories ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_category(category_id: int) -> dict | None:
    db = get_db()
    row = db.execute("SELECT * FROM categories WHERE id = ?", (category_id,)).fetchone()
    return dict(row) if row else None


def create_category(name: str) -> dict:
    db = get_db()
    db.execute("INSERT INTO categories (name) VALUES (?)", (name,))
    db.commit()
    row = db.execute("SELECT * FROM categories WHERE name = ?", (name,)).fetchone()
    return dict(row)


def update_category(category_id: int, name: str) -> dict | None:
    db = get_db()
    db.execute("UPDATE categories SET name = ? WHERE id = ?", (name, category_id))
    db.commit()
    return get_category(category_id)


def delete_category(category_id: int) -> bool:
    db = get_db()
    db.execute("UPDATE projects SET category_id = NULL WHERE category_id = ?", (category_id,))
    cursor = db.execute("DELETE FROM categories WHERE id = ?", (category_id,))
    db.commit()
    return cursor.rowcount > 0


def init_categories():
    db = get_db()
    for name in ("poem", "scifi", "biography"):
        existing = db.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
        if not existing:
            db.execute("INSERT INTO categories (name) VALUES (?)", (name,))
    db.commit()

    cats = {r["name"]: r["id"] for r in db.execute("SELECT id, name FROM categories").fetchall()}

    biography_id = cats.get("biography")
    poem_id = cats.get("poem")
    scifi_id = cats.get("scifi")

    if biography_id:
        db.execute("UPDATE projects SET category_id = ? WHERE category_id IS NULL AND name LIKE '%자서전%'", (biography_id,))
    if poem_id:
        db.execute(
            "UPDATE projects SET category_id = ? WHERE category_id IS NULL AND ("
            "name LIKE '%시 %' OR name LIKE '%시모음%' OR name LIKE '%시인%' "
            "OR name LIKE '%님의침묵%' OR name LIKE '%한용운 시%' OR name LIKE '%시 낭%'"
            ")",
            (poem_id,),
        )
    if scifi_id:
        db.execute(
            "UPDATE projects SET category_id = ? WHERE category_id IS NULL AND ("
            "name LIKE '%투단이%' OR name LIKE '%코딩%포%로%' OR name LIKE '%오파프%'"
            ")",
            (scifi_id,),
        )
    db.commit()
