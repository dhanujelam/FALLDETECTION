"""
alert_manager.py — SQLite Persistence + Webhooks + Email Alerts
================================================================
Handles incident logging to SQLite, webhook notifications (Slack,
Teams, Discord, generic), and optional SMTP email alerts.
All I/O is asynchronous to avoid blocking the video pipeline.
"""

import json
import logging
import smtplib
import sqlite3
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Dict, Any

import requests

from config import cfg

logger = logging.getLogger("surveillance_ai.alerts")


class AlertManager:
    """
    Manages incident persistence and outbound notifications.
    """

    def __init__(self):
        self._db_path = cfg.DB_PATH
        self._db_lock = threading.Lock()
        self._last_alert_time: Dict[str, float] = {}  # event_type -> timestamp
        self._init_db()

    # ── Database ────────────────────────────────────────────────────────────

    def _init_db(self):
        """Initialize the SQLite database and create tables if needed."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL DEFAULT 'INFO',
                        details TEXT,
                        people_count INTEGER DEFAULT 0,
                        acknowledged INTEGER DEFAULT 0,
                        created_at REAL NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incidents_timestamp
                    ON incidents(timestamp DESC)
                """)
                conn.commit()
                conn.close()
                logger.info(f"Incident database initialized: {self._db_path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def log_incident(
        self,
        event_type: str,
        severity: str = "INFO",
        details: str = "",
        people_count: int = 0,
    ) -> Optional[int]:
        """Log an incident to the SQLite database."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute(
                    """INSERT INTO incidents
                       (timestamp, event_type, severity, details, people_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        now.isoformat(),
                        event_type,
                        severity,
                        details,
                        people_count,
                        time.time(),
                    ),
                )
                conn.commit()
                row_id = cursor.lastrowid
                conn.close()

            logger.info(f"Incident logged: [{severity}] {event_type} — {details} (id={row_id})")
            return row_id

        except Exception as e:
            logger.error(f"Failed to log incident: {e}")
            return None

    def get_recent_incidents(self, limit: int = 50) -> list:
        """Retrieve the most recent incidents."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
                rows = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return rows
        except Exception as e:
            logger.error(f"Failed to retrieve incidents: {e}")
            return []
