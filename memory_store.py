"""Local SQLite storage for conversation history and learning progress."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "mentor_memory.sqlite3"


def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def load_messages(limit: int = 100) -> list[dict[str, str]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def save_message(role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        return
    with _connect() as conn:
        conn.execute("INSERT INTO messages(role, content) VALUES (?, ?)", (role, content))
        conn.commit()


def get_progress(key: str, default: str = "") -> str:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM progress WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_progress(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT INTO progress(key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
        """, (key, value))
        conn.commit()


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM progress")
        conn.commit()
