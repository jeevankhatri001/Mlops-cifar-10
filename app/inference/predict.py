import os
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from app.training.model import CIFAR10CNN
from app.ingestion.features import compute_features, CIFAR10_CLASSES

MODEL_PATH = os.getenv("MODEL_PATH", "models/cifar10_cnn.pth")
_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)

_model = None
_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def _get_model():
    global _model
    if _model is None:
        m = CIFAR10CNN()
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _model = m
    return _model


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Features for drift monitoring, computed on the 32x32 image so they are
    # on the same scale as the ingested training baseline.
    arr32 = np.array(image.resize((32, 32)))
    features = compute_features(arr32)

    x = _transform(image).unsqueeze(0)
    model = _get_model()
    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "predicted_label": int(idx.item()),
        "predicted_class": CIFAR10_CLASSES[int(idx.item())],
        "confidence": float(conf.item()),
        "latency_ms": round(latency_ms, 2),
        "features": features,
    }
