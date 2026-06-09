"""Prediction logging + population-level drift reporting."""
from datetime import datetime
from sqlalchemy import text
from app.database.db import engine
from app.monitoring.baseline import compute_baseline, BASELINE_FEATURES

_LOG_SQL = text(
    """
    INSERT INTO predictions
        (requested_at, filename, predicted_label, predicted_class, confidence,
         latency_ms, mean_r, mean_g, mean_b, brightness, contrast,
         model_version, drift_flag)
    VALUES
        (:requested_at, :filename, :predicted_label, :predicted_class, :confidence,
         :latency_ms, :mean_r, :mean_g, :mean_b, :brightness, :contrast,
         :model_version, :drift_flag)
    """
)


def log_prediction(result, filename, model_version="cifar10_cnn", drift_flag=False):
    f = result["features"]
    with engine.begin() as conn:
        conn.execute(_LOG_SQL, {
            "requested_at":    datetime.utcnow(),
            "filename":        filename,
            "predicted_label": result["predicted_label"],
            "predicted_class": result["predicted_class"],
            "confidence":      result["confidence"],
            "latency_ms":      result["latency_ms"],
            "mean_r":          f["mean_r"],
            "mean_g":          f["mean_g"],
            "mean_b":          f["mean_b"],
            "brightness":      f["brightness"],
            "contrast":        f["contrast"],
            "model_version":   model_version,
            "drift_flag":      int(drift_flag),
        })


def drift_report(window=100):
    """Compare the mean feature values of the last `window` predictions to the
    training baseline. Flags a feature if its recent mean is more than 2
    baseline std-devs from the baseline mean.
    """
    baseline = compute_baseline()
    agg = ", ".join(f"avg({f}) AS {f}" for f in BASELINE_FEATURES)
    sql = text(
        f"SELECT {agg}, count(*) AS n FROM "
        f"(SELECT * FROM predictions ORDER BY requested_at DESC LIMIT :w) t"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"w": window}).mappings().first()

    n = row["n"] or 0
    report = {"window": window, "n_predictions": int(n),
              "features": {}, "drift_detected": False}
    if n == 0:
        return report
    for f in BASELINE_FEATURES:
        recent = float(row[f])
        base = baseline[f]
        std = base["std"] or 1e-9
        shift = abs(recent - base["mean"]) / std
        drifting = shift > 2.0
        report["features"][f] = {
            "baseline_mean": round(base["mean"], 2),
            "recent_mean":   round(recent, 2),
            "z_score":       round(shift, 3),
            "drifting":      drifting,
        }
        if drifting:
            report["drift_detected"] = True
    return report
