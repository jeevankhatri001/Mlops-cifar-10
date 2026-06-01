"""Export a few REAL images from the store (Redis) to PNG files, so they can be
uploaded to /predict. Filenames carry the true class for easy comparison.
"""
import os
import numpy as np
from PIL import Image
from sqlalchemy import text
from app.database.db import engine
from app.database.redis_client import get_redis

OUT = "sample_images"
os.makedirs(OUT, exist_ok=True)
r = get_redis()
with engine.connect() as conn:
    rows = conn.execute(text(
        "SELECT image_id, class_name, redis_key FROM images "
        "WHERE split='test' ORDER BY random() LIMIT 8"
    )).fetchall()

for image_id, class_name, key in rows:
    arr = np.frombuffer(r.get(key), dtype=np.uint8).reshape(32, 32, 3)
    path = os.path.join(OUT, f"{class_name}_{image_id}.png")
    Image.fromarray(arr).resize((128, 128), Image.NEAREST).save(path)
    print("saved", path)
