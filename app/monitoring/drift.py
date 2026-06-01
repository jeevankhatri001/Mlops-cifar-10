"""Data-drift detection using z-scores against the training baseline."""

DRIFT_Z_THRESHOLD = 3.0


def zscore_drift(features, baseline, threshold=DRIFT_Z_THRESHOLD):
    """Per-request drift check. Returns (is_drift, z_scores).
    Flags a request if any monitored feature is more than `threshold`
    std-devs from the baseline mean (image looks unlike the training data).
    """
    z_scores = {}
    is_drift = False
    for feat, stats in baseline.items():
        if feat not in features:
            continue
        std = stats["std"] or 1e-9
        z = abs(features[feat] - stats["mean"]) / std
        z_scores[feat] = round(z, 3)
        if z > threshold:
            is_drift = True
    return is_drift, z_scores
