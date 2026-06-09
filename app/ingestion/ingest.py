"""ETL: ingest CIFAR-10 into the analytics store."""
import argparse
from datetime import datetime
import numpy as np
from sqlalchemy import text

from app.database.db import engine, init_db
from app.database.redis_client import get_redis
from app.ingestion.features import compute_features, CIFAR10_CLASSES

REDIS_KEY_FMT = "img:{split}:{idx}"

INSERT_SQL = text(
    """
    INSERT INTO images
        (image_id, split, label, class_name,
         width, height, channels,
         mean_r, mean_g, mean_b, std_r, std_g, std_b,
         brightness, contrast, redis_key, source, ingested_at)
    VALUES
        (:image_id, :split, :label, :class_name,
         :width, :height, :channels,
         :mean_r, :mean_g, :mean_b, :std_r, :std_g, :std_b,
         :brightness, :contrast, :redis_key, :source, :ingested_at)
    """
)


def ingest_samples(samples, split, source="torchvision_cifar10", batch_size=500):
    r = get_redis()
    rows, pipe, total = [], r.pipeline(), 0

    def flush():
        nonlocal rows, pipe
        if not rows:
            return
        with engine.begin() as conn:
            conn.execute(INSERT_SQL, rows)
        pipe.execute()
        rows, pipe = [], r.pipeline()

    now = datetime.utcnow()

    for idx, img, label in samples:
        arr = np.asarray(img, dtype=np.uint8)
        key = REDIS_KEY_FMT.format(split=split, idx=idx)
        pipe.set(key, arr.tobytes())
        rows.append({
            "image_id":    int(idx),
            "split":       split,
            "label":       int(label),
            "class_name":  CIFAR10_CLASSES[int(label)],
            "width":       32,
            "height":      32,
            "channels":    3,
            "redis_key":   key,
            "source":      source,
            "ingested_at": now,
            **compute_features(arr),
        })
        total += 1
        if len(rows) >= batch_size:
            flush()
            print(f"  [{split}] {total} rows flushed ...")

    flush()
    print(f"  [{split}] done — {total} rows total")


def _iter_cifar10(split, limit=None):
    import torchvision
    ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10_data", train=(split == "train"), download=True
    )
    for i, (img, label) in enumerate(ds):
        if limit and i >= limit:
            break
        yield i, img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    init_db()
    for split in args.splits:
        print(f"Ingesting split='{split}' ...")
        ingest_samples(_iter_cifar10(split, args.limit), split=split)


if __name__ == "__main__":
    main()
