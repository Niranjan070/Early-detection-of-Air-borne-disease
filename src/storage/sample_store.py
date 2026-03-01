from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SampleRecord:
    id: int
    farmer_id: str
    date: str  # YYYY-MM-DD
    crop_type: Optional[str]
    exposure_hours: float
    image_path: Optional[str]
    payload: Dict[str, Any]
    created_at: str
    updated_at: str


class SampleStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  farmer_id TEXT NOT NULL,
                  date TEXT NOT NULL,
                  crop_type TEXT,
                  exposure_hours REAL NOT NULL,
                  image_path TEXT,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT (datetime('now')),
                  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                  UNIQUE(farmer_id, date)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_samples_farmer_date
                ON samples(farmer_id, date);
                """
            )

    def upsert(
        self,
        *,
        farmer_id: str,
        date: str,
        crop_type: Optional[str],
        exposure_hours: float,
        image_path: Optional[str],
        payload: Dict[str, Any],
    ) -> SampleRecord:
        payload_json = json.dumps(payload, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO samples (farmer_id, date, crop_type, exposure_hours, image_path, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(farmer_id, date)
                DO UPDATE SET
                  crop_type=excluded.crop_type,
                  exposure_hours=excluded.exposure_hours,
                  image_path=excluded.image_path,
                  payload_json=excluded.payload_json,
                  updated_at=datetime('now');
                """,
                (farmer_id, date, crop_type, float(exposure_hours), image_path, payload_json),
            )

            row = conn.execute(
                "SELECT * FROM samples WHERE farmer_id=? AND date=?",
                (farmer_id, date),
            ).fetchone()

        return self._row_to_record(row)

    def get(self, *, farmer_id: str, date: str) -> Optional[SampleRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM samples WHERE farmer_id=? AND date=?",
                (farmer_id, date),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_recent(self, *, farmer_id: str, limit: int = 14) -> List[SampleRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM samples
                WHERE farmer_id=?
                ORDER BY date DESC
                LIMIT ?
                """,
                (farmer_id, int(limit)),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> SampleRecord:
        payload = json.loads(row["payload_json"])
        return SampleRecord(
            id=int(row["id"]),
            farmer_id=row["farmer_id"],
            date=row["date"],
            crop_type=row["crop_type"],
            exposure_hours=float(row["exposure_hours"]),
            image_path=row["image_path"],
            payload=payload,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
