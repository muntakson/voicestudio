"""Simple JWT-based authentication and daily quota."""

import logging
import os
import time
from datetime import date
from typing import Optional

import bcrypt
import jwt
from fastapi import Header, HTTPException

from app.database import get_db

logger = logging.getLogger(__name__)

SECRET_KEY = os.environ.get("AUTH_SECRET", "")
if not SECRET_KEY:
    raise RuntimeError("AUTH_SECRET environment variable is required. Set it in start.sh or systemd unit.")
ALGORITHM = "HS256"
TOKEN_EXPIRE = 60 * 60 * 24 * 30  # 30 days

DAILY_QUOTA_KRW = 250
KRW_PER_USD = 1380


def _hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_pw(password: str, hashed: str) -> bool:
    if hashed.startswith("$2b$") or hashed.startswith("$2a$"):
        return bcrypt.checkpw(password.encode(), hashed.encode())
    # Legacy SHA-256 hash — verify then upgrade
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest() == hashed


def _maybe_upgrade_hash(username: str, password: str, current_hash: str):
    """Upgrade legacy SHA-256 hashes to bcrypt on successful login."""
    if current_hash.startswith("$2b$") or current_hash.startswith("$2a$"):
        return
    new_hash = _hash_pw(password)
    db = get_db()
    db.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username))
    db.commit()
    logger.info("Upgraded password hash for user '%s' to bcrypt", username)


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
    db.execute("""
        CREATE TABLE IF NOT EXISTS daily_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            usage_date TEXT NOT NULL,
            service TEXT NOT NULL,
            cost_usd REAL NOT NULL DEFAULT 0,
            cost_krw REAL NOT NULL DEFAULT 0,
            detail TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_usage_user_date ON daily_usage(username, usage_date)
    """)
    db.commit()

    admin_pw = os.environ.get("ADMIN_PASSWORD", "")
    default_user_pw = os.environ.get("DEFAULT_USER_PASSWORD", "")

    existing = db.execute("SELECT username FROM users").fetchall()
    names = {r["username"] for r in existing}
    if "admin" not in names and admin_pw:
        db.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, datetime('now'))",
            ("admin", _hash_pw(admin_pw), "admin"),
        )
    if "sonny" not in names and default_user_pw:
        db.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, datetime('now'))",
            ("sonny", _hash_pw(default_user_pw), "user"),
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
    remaining = get_remaining_quota(username)
    return {"username": username, "role": "user", "token": token, "quota_remaining_krw": remaining}


def signin(username: str, password: str) -> dict:
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not _check_pw(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    _maybe_upgrade_hash(username, password, row["password_hash"])
    token = _create_token(row["username"], row["role"])
    remaining = get_remaining_quota(row["username"]) if row["role"] != "admin" else -1
    return {"username": row["username"], "role": row["role"], "token": token, "quota_remaining_krw": remaining}


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


# ---------------------------------------------------------------------------
# Quota
# ---------------------------------------------------------------------------

def get_today_usage_krw(username: str) -> float:
    db = get_db()
    today = date.today().isoformat()
    row = db.execute(
        "SELECT COALESCE(SUM(cost_krw), 0) as total FROM daily_usage WHERE username = ? AND usage_date = ?",
        (username, today),
    ).fetchone()
    return row["total"] if row else 0


def get_remaining_quota(username: str) -> float:
    used = get_today_usage_krw(username)
    return round(DAILY_QUOTA_KRW - used, 2)


def record_usage(username: str, service: str, cost_usd: float, detail: str = ""):
    cost_krw = round(cost_usd * KRW_PER_USD, 2)
    db = get_db()
    today = date.today().isoformat()
    db.execute(
        "INSERT INTO daily_usage (username, usage_date, service, cost_usd, cost_krw, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (username, today, service, round(cost_usd, 6), cost_krw, detail),
    )
    db.commit()


def check_quota(user: Optional[dict], estimated_cost_usd: float = 0) -> Optional[dict]:
    """Check if user has enough quota. Returns user dict or raises 429."""
    if not user:
        return None
    if user["role"] == "admin":
        return user
    remaining = get_remaining_quota(user["username"])
    estimated_krw = estimated_cost_usd * KRW_PER_USD
    if remaining <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"일일 사용량 한도 초과 (₩{DAILY_QUOTA_KRW}/일). 내일 다시 시도해주세요. Daily quota exceeded."
        )
    return user
