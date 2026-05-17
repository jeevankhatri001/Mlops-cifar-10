CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255),
    run_id VARCHAR(255),
    accuracy FLOAT,
    loss FLOAT,
    epochs INTEGER,
    learning_rate FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
