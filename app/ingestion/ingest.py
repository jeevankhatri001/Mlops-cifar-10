"""ETL: ingest CIFAR-10 into the analytics store.

  Extract   - pull images from torchvision (the raw source)
  Transform - compute feature variables per image
  Load      - metadata + features -> PostgreSQL ('images' table)
              raw pixel array      -> Redis        (key 'img:{split}:{idx}')

Run:
    python -m app.ingestion.ingest                 # full train + test
    python -m app.ingestion.ingest --limit 200     # quick smoke test
    python -m app.ingestion.ingest --splits test   # one split only
"""
import argparse
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
         mean_r, mean_g, mean_b, std_r, std_g, std_b,
         brightness, contrast, redis_key, source)
    VALUES
        (:image_id, :split, :label, :class_name,
         :mean_r, :mean_g, :mean_b, :std_r, :std_g, :std_b,
         :brightness, :contrast, :redis_key, :source)
    ON CONFLICT (split, image_id) DO NOTHING
    """
)


def ingest_samples(samples, split, source="torchvision_cifar10", batch_size=500):
    """Load an iterable of (idx, image_uint8_array, label) into the store."""
    r = get_redis()
    rows, pipe, total = [], r.pipeline(), 0

    def flush():
        nonlocal rows, pipe
        if not rows:
            return
        with engine.begin() as conn:        # metadata -> Postgres
            conn.execute(INSERT_SQL, rows)
        pipe.execute()                       # pixel arrays -> Redis
        rows, pipe = [], r.pipeline()

    for idx, img, label in samples:
        arr = np.asarray(img, dtype=np.uint8)
        key = REDIS_KEY_FMT.format(split=split, idx=idx)
        pipe.set(key, arr.tobytes())
        rows.append({
            "image_id": int(idx), "split": split, "label": int(label),
            "class_name": CIFAR10_CLASSES[int(label)], "redis_key": key,
            "source": source, **compute_features(arr),
        })
        total += 1
        if len(rows) >= batch_size:
            flush()
    flush()
    return total


def torchvision_samples(split, limit=None):
    """Extract: yield (idx, HxWx3 uint8 array, label) from torchvision CIFAR-10."""
    from torchvision import datasets
    ds = datasets.CIFAR10(root="./data", train=(split == "train"), download=True)
    for idx in range(len(ds)):
        if limit is not None and idx >= limit:
            break
        img, label = ds[idx]                 # img is a PIL.Image
        yield idx, np.array(img), label


def main():
    ap = argparse.ArgumentParser(description="Ingest CIFAR-10 into Postgres + Redis")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max images per split (for quick tests)")
    args = ap.parse_args()

    init_db()
    for split in args.splits:
        n = ingest_samples(torchvision_samples(split, args.limit), split)
        print(f"Ingested {n} images for split='{split}'")


if __name__ == "__main__":
    main()
