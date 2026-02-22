from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

_DB_PATH: Optional[str] = None


def init_db(db_path: str) -> None:
    global _DB_PATH
    _DB_PATH = db_path

    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                culture TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    if not _DB_PATH:
        raise RuntimeError("DB is not initialized")
    return sqlite3.connect(_DB_PATH)


def add_prediction(filename: str, label: str, confidence: float, culture: Optional[str] = None) -> int:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (created_at, filename, label, confidence, culture) VALUES (?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(timespec="seconds") + "Z", filename, label, confidence, culture),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_history(limit: int = 10) -> List[Dict]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, created_at, filename, label, confidence, culture FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "filename": r[2],
                "label": r[3],
                "confidence": r[4],
                "culture": r[5],
            }
            for r in rows
        ]
    finally:
        conn.close()


def clear_history() -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    finally:
        conn.close()
