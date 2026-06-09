CREATE TABLE IF NOT EXISTS images (
    image_id    INT          NOT NULL,
    split       VARCHAR(10)  NOT NULL,
    label       INT          NOT NULL,
    class_name  VARCHAR(32)  NOT NULL,
    width       INT,
    height      INT,
    channels    INT,
    mean_r      FLOAT,
    mean_g      FLOAT,
    mean_b      FLOAT,
    std_r       FLOAT,
    std_g       FLOAT,
    std_b       FLOAT,
    brightness  FLOAT,
    contrast    FLOAT,
    redis_key   VARCHAR(64)  NOT NULL,
    source      VARCHAR(64)  NOT NULL,
    ingested_at DATETIME
) ENGINE=Columnstore DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS experiments (
    id              BIGINT,
    experiment_name VARCHAR(255),
    run_id          VARCHAR(255),
    accuracy        DOUBLE,
    loss            DOUBLE,
    epochs          INT,
    learning_rate   DOUBLE,
    batch_size      INT,
    created_at      DATETIME
) ENGINE=Columnstore DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS predictions (
    id              BIGINT,
    requested_at    DATETIME,
    filename        VARCHAR(255),
    predicted_label INT,
    predicted_class VARCHAR(32),
    confidence      FLOAT,
    latency_ms      FLOAT,
    mean_r          FLOAT,
    mean_g          FLOAT,
    mean_b          FLOAT,
    brightness      FLOAT,
    contrast        FLOAT,
    model_version   VARCHAR(64),
    drift_flag      TINYINT(1)
) ENGINE=Columnstore DEFAULT CHARSET=utf8mb4
