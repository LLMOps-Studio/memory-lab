import random
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path


def generate_synthetic_feedback(db_path: str = "feedback.db", count: int = 50):
    """
    Simulates user interactions and injects synthetic feedback ratings (1 or -1)
    into the SQLite database to test the MLflow logging pipeline.
    """
    db_file = Path(db_path)

    # Initialize DB if not exists (though the UI should have created it)
    with sqlite3.connect(db_file) as conn:
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

        for i in range(count):
            # Simulate a 75% success rate for the retrieval system
            rating = 1 if random.random() > 0.25 else -1

            # Scatter timestamps over the last 30 days
            past_date = datetime.now(UTC) - timedelta(days=random.randint(0, 30))

            conn.execute(
                "INSERT INTO memory_feedback (id, session_id, user_message, ai_response, rating, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    f"sim_session_{i:03d}",
                    "Simulated fact retrieval",
                    "Simulated AI memory response",
                    rating,
                    past_date,
                ),
            )

    print(f"✅ Successfully injected {count} synthetic feedback records into {db_path}")


if __name__ == "__main__":
    generate_synthetic_feedback()
