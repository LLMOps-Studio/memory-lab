import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

class FeedbackDB:
    """
    Manages local SQLite storage for user feedback on memory retrieval quality.
    This data will later be aggregated and logged to MLflow to track model drift.
    """
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Creates the feedback table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_feedback (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_message TEXT,
                    ai_response TEXT,
                    rating INTEGER,
                    timestamp DATETIME
                )
            """)

    def log_feedback(self, session_id: str, user_msg: str, ai_response: str, rating: int):
        """Logs a single user interaction and its explicit rating (1 for 👍, -1 for 👎)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory_feedback (id, session_id, user_message, ai_response, rating, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), session_id, user_msg, ai_response, rating, datetime.now())
            )
        print(f"[Feedback] Logged rating {rating} for session {session_id}")