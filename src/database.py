import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


DB_PATH = "training/results/emotion_app.db"


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False)


def init_db(db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            recording_path TEXT,
            notes TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            rms REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """
    )
    conn.commit()
    conn.close()


def create_session(session_id: str, db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO sessions(session_id, started_at) VALUES(?, ?)",
        (session_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def close_session(session_id: str, recording_path: Optional[str], db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET ended_at=?, recording_path=? WHERE session_id=?",
        (datetime.utcnow().isoformat(), recording_path, session_id),
    )
    conn.commit()
    conn.close()


def insert_prediction(session_id: str, emotion: str, confidence: float, rms: float, db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions(session_id, ts_utc, emotion, confidence, rms) VALUES(?, ?, ?, ?, ?)",
        (session_id, datetime.utcnow().isoformat(), emotion, float(confidence), float(rms)),
    )
    conn.commit()
    conn.close()


def list_sessions(limit: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT session_id, started_at, ended_at, recording_path
        FROM sessions
        ORDER BY started_at DESC
        LIMIT ?
        """,
        conn,
        params=(int(limit),),
    )
    conn.close()
    return df


def get_predictions(session_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT ts_utc, emotion, confidence, rms
        FROM predictions
        WHERE session_id = ?
        ORDER BY ts_utc ASC
        """,
        conn,
        params=(session_id,),
    )
    conn.close()
    return df
