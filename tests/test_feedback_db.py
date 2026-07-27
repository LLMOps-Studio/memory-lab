import sqlite3

from memory_lab.memory.feedback_db import FeedbackDB

# There's no HTTP endpoint for feedback storage -- FeedbackDB is only wired
# into the Streamlit UI (app.py), not memory_lab.api -- so this exercises
# the actual feedback-storage logic directly against a temp SQLite file.


def test_log_feedback_persists_a_row(tmp_path):
    db_path = tmp_path / "feedback.db"
    db = FeedbackDB(db_path=str(db_path))

    db.log_feedback("session-1", "I love hiking", "Noted!", 1)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id, user_message, ai_response, rating FROM memory_feedback"
        ).fetchall()

    assert rows == [("session-1", "I love hiking", "Noted!", 1)]


def test_log_feedback_supports_multiple_rows_and_negative_rating(tmp_path):
    db_path = tmp_path / "feedback.db"
    db = FeedbackDB(db_path=str(db_path))

    db.log_feedback("session-1", "msg one", "resp one", 1)
    db.log_feedback("session-1", "msg two", "resp two", -1)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM memory_feedback").fetchone()[0]
        negative = conn.execute(
            "SELECT COUNT(*) FROM memory_feedback WHERE rating = -1"
        ).fetchone()[0]

    assert count == 2
    assert negative == 1


def test_init_db_is_idempotent(tmp_path):
    """Constructing FeedbackDB against an existing db file must not error."""
    db_path = tmp_path / "feedback.db"
    FeedbackDB(db_path=str(db_path))
    FeedbackDB(db_path=str(db_path))  # should not raise
