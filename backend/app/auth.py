"""Simple JWT-based authentication."""

import hashlib
import os
import time
from typing import Optional

import jwt
from fastapi import Header, HTTPException

from app.database import get_db

SECRET_KEY = os.environ.get("AUTH_SECRET", "voicestudio-secret-key-2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE = 60 * 60 * 24 * 30  # 30 days


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_users():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)
    db.commit()
    existing = db.execute("SELECT username FROM users").fetchall()
    names = {r["username"] for r in existing}
    if "admin" not in names:
        db.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, datetime('now'))",
            ("admin", _hash_pw("intel8051"), "admin"),
        )
    if "sonny" not in names:
        db.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, datetime('now'))",
            ("sonny", _hash_pw("q1"), "user"),
        )
    db.commit()


def signup(username: str, password: str) -> dict:
    db = get_db()
    existing = db.execute("SELECT username FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")
    db.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, datetime('now'))",
        (username, _hash_pw(password), "user"),
    )
    db.commit()
    token = _create_token(username, "user")
    return {"username": username, "role": "user", "token": token}


def signin(username: str, password: str) -> dict:
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row or row["password_hash"] != _hash_pw(password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _create_token(row["username"], row["role"])
    return {"username": row["username"], "role": row["role"], "token": token}


def _create_token(username: str, role: str) -> str:
    payload = {"sub": username, "role": role, "exp": int(time.time()) + TOKEN_EXPIRE}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"], "role": payload.get("role", "user")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_auth(authorization: Optional[str] = Header(default=None)) -> dict:
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def assign_orphan_projects():
    db = get_db()
    cursor = db.execute("PRAGMA table_info(projects)")
    cols = {row[1] for row in cursor.fetchall()}
    if "owner" not in cols:
        db.execute("ALTER TABLE projects ADD COLUMN owner TEXT DEFAULT 'admin'")
    db.execute("UPDATE projects SET owner = 'admin' WHERE owner IS NULL OR owner = ''")
    db.commit()
