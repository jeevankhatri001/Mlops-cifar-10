import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def compute_features(image):
    """Compute scalar feature variables from an HxWxC uint8 RGB image."""
    arr = np.asarray(image)
    if arr.ndim == 2:                      # grayscale -> 3 channels
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.astype(np.float64)
    flat = arr.reshape(-1, arr.shape[-1])  # (pixels, channels)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    gray = arr.mean(axis=-1)
    return {
        "mean_r": float(means[0]), "mean_g": float(means[1]), "mean_b": float(means[2]),
        "std_r": float(stds[0]), "std_g": float(stds[1]), "std_b": float(stds[2]),
        "brightness": float(arr.mean()),
        "contrast": float(gray.std()),
    }
