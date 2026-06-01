import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File
from sqlalchemy import text

from app.inference.predict import predict_image
from app.monitoring.baseline import compute_baseline
from app.monitoring.drift import zscore_drift
from app.monitoring.monitor import log_prediction, drift_report
from app.database.db import engine

app = FastAPI(title="CIFAR-10 MLOps API",
              description="Inference + monitoring API", version="2.0.0")

MODEL_VERSION = os.path.basename(os.getenv("MODEL_PATH", "models/cifar10_cnn.pth"))
_baseline = None


def get_baseline():
    global _baseline
    if _baseline is None:
        _baseline = compute_baseline()
    return _baseline


@app.get("/")
def root():
    return {"message": "CIFAR-10 MLOps API is running", "version": "2.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = predict_image(tmp_path)
    finally:
        os.remove(tmp_path)

    is_drift, z_scores = zscore_drift(result["features"], get_baseline())
    log_prediction(result, file.filename, model_version=MODEL_VERSION, drift_flag=is_drift)

    return {
        "filename": file.filename,
        "prediction": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "latency_ms": result["latency_ms"],
        "drift": {"flagged": is_drift, "z_scores": z_scores},
    }


@app.get("/monitoring/stats")
def monitoring_stats():
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM predictions")).scalar()
        drift_n = conn.execute(text("SELECT count(*) FROM predictions WHERE drift_flag")).scalar()
        avg_conf = conn.execute(text("SELECT avg(confidence) FROM predictions")).scalar()
        by_class = conn.execute(text(
            "SELECT predicted_class, count(*) c FROM predictions "
            "GROUP BY predicted_class ORDER BY c DESC"
        )).fetchall()
    return {
        "total_predictions": int(total or 0),
        "drift_flagged": int(drift_n or 0),
        "avg_confidence": round(float(avg_conf), 4) if avg_conf is not None else None,
        "by_class": {r[0]: r[1] for r in by_class},
    }


@app.get("/monitoring/drift")
def monitoring_drift(window: int = 100):
    return drift_report(window=window)
