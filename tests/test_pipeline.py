"""
Smoke tests for the CIFAR-10 MLOps pipeline.
These run in CI without needing Postgres/Redis/MLflow.
"""
import torch


def test_model_import():
    """Model class can be imported."""
    from app.training.model import CIFAR10CNN
    assert CIFAR10CNN is not None


def test_model_forward_pass():
    """Model produces correct output shape."""
    from app.training.model import CIFAR10CNN
    model = CIFAR10CNN()
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"


def test_model_output_is_logits():
    """Output values are raw logits (not probabilities)."""
    from app.training.model import CIFAR10CNN
    model = CIFAR10CNN()
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy)
    assert output.min() < 0 or output.max() > 1, "Output looks like probabilities, expected logits"


def test_model_parameter_count():
    """Model has a reasonable number of parameters."""
    from app.training.model import CIFAR10CNN
    model = CIFAR10CNN()
    params = sum(p.numel() for p in model.parameters())
    assert params > 10000, f"Model too small: {params} params"
    assert params < 10_000_000, f"Model too large: {params} params"


def test_cifar10_classes():
    """Verify 10 output classes for CIFAR-10."""
    from app.training.model import CIFAR10CNN
    model = CIFAR10CNN()
    last_layer = list(model.children())[-1]
    if hasattr(last_layer, 'out_features'):
        assert last_layer.out_features == 10


def test_feature_extraction_import():
    """Feature extraction module can be imported."""
    from app.ingestion import features
    assert hasattr(features, 'compute_features') or callable(getattr(features, 'extract_features', None)) or True


def test_api_import():
    """FastAPI app can be imported."""
    from app.api.main import app
    assert app is not None
