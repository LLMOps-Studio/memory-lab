import os
import sqlite3
from pathlib import Path

import mlflow


def run_feedback_analysis(db_path: str = "feedback.db"):
    """
    Reads all user feedback from the local SQLite database,
    calculates memory retrieval success metrics, and logs them to MLflow.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"❌ Database {db_path} not found. Generate data first.")
        return

    print("🔍 Analyzing local feedback database...")

    # 1. Query SQLite
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memory_feedback")
        total_votes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memory_feedback WHERE rating = 1")
        positive_votes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memory_feedback WHERE rating = -1")
        negative_votes = cursor.fetchone()[0]

    if total_votes == 0:
        print("⚠️ No feedback data available to analyze.")
        return

    # Calculate metrics
    positive_rate = positive_votes / total_votes

    # 2. Log to Central MLflow Platform
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_DEFAULT_EXPERIMENT", "memory_lab_feedback")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"📡 Connecting to MLflow at {tracking_uri}...")
    with mlflow.start_run(run_name="weekly_memory_performance"):
        mlflow.log_metrics(
            {
                "total_feedback_count": total_votes,
                "positive_feedback_count": positive_votes,
                "negative_feedback_count": negative_votes,
                "positive_feedback_rate": positive_rate,
            }
        )

    # 3. Output Results
    print("\n📊 Memory System Performance Report")
    print("-" * 35)
    print(f"Total User Votes   : {total_votes}")
    print(f"Accurate Retrievals: {positive_votes}")
    print(f"Hallucinations     : {negative_votes}")
    print(f"Success Rate       : {positive_rate * 100:.1f}%")
    print("-" * 35)
    print("✅ Metrics successfully committed to MLflow!")


if __name__ == "__main__":
    run_feedback_analysis()
