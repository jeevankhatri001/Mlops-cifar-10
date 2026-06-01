"""Data loading for the CIFAR-10 pipeline.

  get_dataloaders()            - download direct from torchvision (original)
  get_dataloaders_from_store() - read INGESTED data from the analytics store
                                 (Postgres labels + Redis pixels)
"""
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def _train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def get_dataloaders(batch_size=64):
    train_dataset = datasets.CIFAR10(root="./data", train=True,
                                     download=True, transform=_train_transform())
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=True, transform=_test_transform())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


def load_split_from_store(split):
    """Return (images Nx32x32x3 uint8, labels N int64) for a split."""
    from sqlalchemy import text
    from app.database.db import engine
    from app.database.redis_client import get_redis

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT image_id, label, redis_key FROM images "
            "WHERE split = :s ORDER BY image_id"
        ), {"s": split}).fetchall()

    if not rows:
        raise RuntimeError(
            f"No images found for split='{split}'. "
            f"Run ingestion first: python -m app.ingestion.ingest"
        )

    keys = [r[2] for r in rows]
    labels = np.array([r[1] for r in rows], dtype=np.int64)
    raws = get_redis().mget(keys)
    images = np.stack([
        np.frombuffer(b, dtype=np.uint8).reshape(32, 32, 3) for b in raws
    ])
    return images, labels


class StoreDataset(Dataset):
    """Torch Dataset over in-memory uint8 images from the analytics store."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self.labels[idx])


def get_dataloaders_from_store(batch_size=64):
    """DataLoaders sourced from the ingested analytics store."""
    train_imgs, train_lbls = load_split_from_store("train")
    test_imgs, test_lbls = load_split_from_store("test")
    train_ds = StoreDataset(train_imgs, train_lbls, _train_transform())
    test_ds = StoreDataset(test_imgs, test_lbls, _test_transform())
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader
