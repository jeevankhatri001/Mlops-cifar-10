-- =====================================================================
-- CMP5366 MLOps Pipeline - Analytics store schema (PostgreSQL)
--   images      : one row of computed features per ingested image
--   experiments : one row per training run (lineage / monitoring)
--   predictions : one row per live inference request (drift monitoring)
-- Raw pixel arrays live in Redis, referenced here by redis_key.
-- =====================================================================

CREATE TABLE IF NOT EXISTS images (
    image_id    INTEGER      NOT NULL,        -- index within the split
    split       VARCHAR(10)  NOT NULL,        -- 'train' | 'test'
    label       INTEGER      NOT NULL,        -- target variable (0-9)
    class_name  VARCHAR(32)  NOT NULL,        -- human-readable target
    width       INTEGER      NOT NULL DEFAULT 32,
    height      INTEGER      NOT NULL DEFAULT 32,
    channels    INTEGER      NOT NULL DEFAULT 3,
    mean_r      REAL,                          -- feature: per-channel mean
    mean_g      REAL,
    mean_b      REAL,
    std_r       REAL,                          -- feature: per-channel spread
    std_g       REAL,
    std_b       REAL,
    brightness  REAL,                          -- feature: overall mean intensity
    contrast    REAL,                          -- feature: grayscale std-dev
    redis_key   VARCHAR(64)  NOT NULL,         -- pointer to pixel array in Redis
    source      VARCHAR(64)  NOT NULL DEFAULT 'torchvision_cifar10',
    ingested_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (split, image_id)
);
CREATE INDEX IF NOT EXISTS idx_images_label ON images(label);
CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);

CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255),
    run_id          VARCHAR(255),
    accuracy        FLOAT,
    loss            FLOAT,
    epochs          INTEGER,
    learning_rate   FLOAT,
    batch_size      INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    requested_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    filename        VARCHAR(255),
    predicted_label INTEGER,
    predicted_class VARCHAR(32),
    confidence      REAL,
    latency_ms      REAL,
    mean_r          REAL,
    mean_g          REAL,
    mean_b          REAL,
    brightness      REAL,
    contrast        REAL,
    model_version   VARCHAR(64),
    drift_flag      BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(requested_at);
