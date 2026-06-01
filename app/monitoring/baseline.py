"""Baseline feature distribution, computed from the INGESTED training data.
This is the reference distribution that live inference is compared against
to detect data drift.
"""
from sqlalchemy import text
from app.database.db import engine

BASELINE_FEATURES = ("brightness", "contrast", "mean_r", "mean_g", "mean_b")


def compute_baseline(split="train"):
    """Return {feature: {'mean': x, 'std': y}} over the training images."""
    cols = ", ".join(
        f"avg({f}) AS {f}_mean, stddev_samp({f}) AS {f}_std"
        for f in BASELINE_FEATURES
    )
    sql = text(f"SELECT {cols} FROM images WHERE split = :s")
    with engine.connect() as conn:
        row = conn.execute(sql, {"s": split}).mappings().first()
    return {
        f: {"mean": float(row[f"{f}_mean"]), "std": float(row[f"{f}_std"])}
        for f in BASELINE_FEATURES
    }
