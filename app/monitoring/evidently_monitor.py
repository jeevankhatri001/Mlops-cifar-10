"""
Evidently AI drift detection for the CIFAR-10 MLOps pipeline.
Compares training baseline features (images table) against
recent prediction features (predictions table).

Usage:
    python -m app.monitoring.evidently_monitor

Generates: evidently_drift_report.html
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from evidently import Report
from evidently.presets import DataDriftPreset

# MariaDB connection
DB_USER = os.getenv("MARIADB_USER", "mlops")
DB_PASS = os.getenv("MARIADB_PASSWORD", "mlops123")
DB_HOST = os.getenv("MARIADB_HOST", "localhost")
DB_PORT = os.getenv("MARIADB_PORT", "3306")
DB_NAME = os.getenv("MARIADB_DB", "mlops_db")

DATABASE_URL = (
    f"mysql+pymysql://{quote_plus(DB_USER)}:{quote_plus(DB_PASS)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

FEATURE_COLS = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "brightness", "contrast"]


def get_reference_data(engine):
    cols = ", ".join(FEATURE_COLS)
    query = text(f"SELECT {cols} FROM images WHERE split = 'train' LIMIT 5000")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    print(f"Reference data: {len(df)} rows")
    return df


def get_current_data(engine):
    cols = ", ".join(FEATURE_COLS)
    try:
        query = text(f"SELECT {cols} FROM predictions ORDER BY requested_at DESC LIMIT 500")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        if len(df) > 0:
            print(f"Current data: {len(df)} rows (from predictions)")
            return df
    except Exception as e:
        print(f"Predictions table issue: {e}")

    print("Falling back to test split...")
    query = text(f"SELECT {cols} FROM images WHERE split = 'test' LIMIT 2000")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    print(f"Current data: {len(df)} rows (test split)")
    return df


def generate_drift_report():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    print("Loading reference data (training set)...")
    reference = get_reference_data(engine)

    print("Loading current data...")
    current = get_current_data(engine)

    if reference.empty or current.empty:
        print("ERROR: Not enough data.")
        return None

    print("Generating Evidently drift report...")
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    output_path = "evidently_drift_report.html"
    snapshot.save_html(output_path)
    print(f"\nReport saved to: {output_path}")

    return snapshot.dict()


if __name__ == "__main__":
    result = generate_drift_report()
    if result:
        print("Drift report generated successfully!")
        print("Open evidently_drift_report.html in your browser.")
