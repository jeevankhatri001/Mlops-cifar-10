# CIFAR-10 MLOps Pipeline

An end-to-end MLOps pipeline for CIFAR-10 image classification, built as a CMP5366 coursework project. The system implements a complete data-analytics and machine-learning pipeline with polyglot persistence (PostgreSQL + Redis), automated orchestration, live inference with drift monitoring, and exploratory data analysis.

## Tech Stack

| Component | Technology | Role in Pipeline |
|---|---|---|
| Deep Learning | PyTorch | CNN model training and inference |
| Dataset | CIFAR-10 (60,000 images) | Supervised image classification |
| Relational Store | PostgreSQL 16 | Analytics store — image features, experiments, predictions (OBT design) |
| Key-Value Store | Redis 7 | Fast pixel-array storage and retrieval for training/inference |
| Experiment Tracking | MLflow | Hyperparameter, metric, and artifact logging |
| API Framework | FastAPI | REST inference endpoint with Swagger UI |
| Monitoring | Custom (z-score drift) | Per-request and population-level data-drift detection |
| Orchestration | Apache Airflow | DAG-based pipeline automation |
| Containerisation | Docker Compose | Service management for Postgres, Redis, Airflow |
| Deployment | AWS EC2 (Ubuntu) | Cloud hosting |

## Project Structure

```
cifar10-mlops/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                # FastAPI app — /predict, /monitoring/stats, /monitoring/drift
│   ├── database/
│   │   ├── db.py                  # SQLAlchemy engine + init_db()
│   │   ├── redis_client.py        # Redis connection pool
│   │   └── schema.sql             # Analytics store schema (images, experiments, predictions)
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py             # Lazy model loading, prediction + feature extraction
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── features.py            # Per-image feature engineering (brightness, contrast, RGB means)
│   │   └── ingest.py              # ETL: torchvision → Postgres (features) + Redis (pixels)
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── baseline.py            # Training-set feature distribution (drift reference)
│   │   ├── drift.py               # Z-score drift detection
│   │   └── monitor.py             # Prediction logging + population drift report
│   └── training/
│       ├── dataset.py             # DataLoaders — torchvision or store-backed
│       ├── evaluate.py            # Model evaluation
│       ├── model.py               # CNN architecture (BatchNorm + Dropout)
│       └── train.py               # Training loop — reads from store, logs to MLflow + DB
├── airflow/
│   ├── dags/
│   │   └── cifar10_training_dag.py  # check_env → ingest → train → verify → complete
│   ├── docker-compose-airflow.yml
│   └── Dockerfile                 # CPU-only torch + pipeline deps
├── notebooks/
│   ├── build_eda_notebook.py      # Generates eda.ipynb
│   └── eda.ipynb                  # Task 3 EDA — connects to Postgres analytics store
├── scripts/
│   └── export_samples.py          # Pull real CIFAR images from Redis → PNG for testing
├── docker-compose.yml             # Postgres + Redis for the app
├── requirements.txt
├── .env                           # Credentials (gitignored)
├── .gitignore
└── README.md
```

## Architecture

### Storage Design (Polyglot Persistence)

The pipeline uses a **denormalised One Big Table (OBT)** in PostgreSQL for structured, queryable metadata and a **Redis key-value store** for raw pixel arrays:

| Store | Table / Key Pattern | Purpose |
|---|---|---|
| PostgreSQL | `images` | Per-image feature columns (brightness, contrast, RGB means/stds, label, split) for EDA and drift baseline |
| PostgreSQL | `experiments` | Training run log (accuracy, loss, hyperparams, MLflow run ID) |
| PostgreSQL | `predictions` | Live inference log (predicted class, confidence, latency, input features, drift flag) |
| Redis | `img:{split}:{idx}` | Raw 32×32×3 uint8 pixel arrays for fast training/inference retrieval |

### Pipeline Stages

```
1. Data Ingestion     torchvision → feature extraction → Postgres + Redis
2. Data Preprocessing normalisation, augmentation (transforms in DataLoader)
3. Model Development  CNN training from the store, logged to MLflow + experiments table
4. Model Deployment   serialised .pth, served via FastAPI /predict, Dockerised
5. Model Monitoring   every prediction logged; z-score drift detection per request + population level
```

### Airflow DAG

```
check_environment → ingest_data → train_model → verify_model_artifact → pipeline_complete
```

Tasks run inside the Airflow worker container and reach the app's Postgres/Redis/MLflow on the host via `host.docker.internal`.

## Model Performance

| Metric | Value |
|---|---|
| Architecture | 3-block CNN (BatchNorm + Dropout) |
| Test Accuracy | ~77% (10 epochs) |
| Optimiser | Adam (lr=0.001) |
| Batch Size | 64 |
| Data Source | Analytics store (Postgres labels + Redis pixels) |

## Setup

### Prerequisites

- AWS EC2 Ubuntu instance (t2.medium or higher)
- Docker and Docker Compose installed
- Python 3.9+ with venv
- Security group ports open: 22 (SSH), 5000 (MLflow), 8000 (FastAPI), 8080 (Airflow)

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/jeevankhatri001/Mlops-cifar-10.git
cd cifar10-mlops
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` in the repo root (gitignored — never commit this):

```bash
POSTGRES_USER=jeevan
POSTGRES_PASSWORD=your_password_here
POSTGRES_DB=mlops_db
POSTGRES_PORT=5432
POSTGRES_HOST=localhost
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Start Storage Services

```bash
docker compose up -d postgres redis
```

### 4. Initialise the Database Schema

```bash
python -m app.database.db
# Expected: Tables in database: ['experiments', 'images', 'predictions']
```

### 5. Ingest CIFAR-10 into the Analytics Store

```bash
python -m app.ingestion.ingest
# Ingests 50,000 train + 10,000 test images (features → Postgres, pixels → Redis)
```

### 6. Start MLflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000 &
```

### 7. Train the Model

```bash
python -m app.training.train
# Reads from the store, logs to MLflow + experiments table, saves models/cifar10_cnn.pth
```

### 8. Run the API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### 9. Run the EDA Notebook

```bash
python notebooks/build_eda_notebook.py
jupyter nbconvert --to html --execute notebooks/eda.ipynb
# Produces notebooks/eda.html with all plots embedded
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| POST | `/predict` | Upload an image → prediction + confidence + drift check |
| GET | `/monitoring/stats` | Total predictions, drift count, class distribution |
| GET | `/monitoring/drift?window=100` | Population drift report vs training baseline |

Swagger UI: `http://<EC2_IP>:8000/docs`

## Monitoring and Drift Detection

Every `/predict` call:

1. Computes image features (brightness, contrast, RGB channel means) on the resized 32×32 input
2. Runs a **z-score drift check** against the training baseline (flags if any feature > 3σ from the mean)
3. Logs the prediction, features, confidence, latency, and drift flag to the `predictions` table

The `/monitoring/drift` endpoint aggregates the last N predictions and flags **population-level drift** when the recent feature means shift more than 2σ from baseline.

## Airflow Setup

```bash
cd airflow
cp ../.env .env
echo "AIRFLOW_UID=$(id -u)" >> .env
mkdir -p plugins logs
docker compose -f docker-compose-airflow.yml build
docker compose -f docker-compose-airflow.yml up -d
```

Access Airflow UI: `http://<EC2_IP>:8080` (admin / admin)

Trigger the pipeline:

```bash
docker exec -it airflow_scheduler airflow dags unpause cifar10_training_pipeline
docker exec -it airflow_scheduler airflow dags trigger cifar10_training_pipeline
```

## Useful Commands

```bash
# Check containers
docker ps

# Verify ingested data
docker exec -it cifar10_postgres psql -U jeevan -d mlops_db \
  -c "SELECT split, count(*) FROM images GROUP BY split;"

# Check experiment history
docker exec -it cifar10_postgres psql -U jeevan -d mlops_db \
  -c "SELECT id, run_id, round(accuracy::numeric,2) AS acc, epochs, created_at FROM experiments ORDER BY id DESC LIMIT 5;"

# Check prediction log
docker exec -it cifar10_postgres psql -U jeevan -d mlops_db \
  -c "SELECT id, predicted_class, round(confidence::numeric,3) AS conf, drift_flag, requested_at FROM predictions ORDER BY id DESC LIMIT 10;"

# Redis key count
docker exec -it cifar10_redis redis-cli DBSIZE
```

## CIFAR-10 Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Author

**Jeevan Khatri**
Birmingham City University — CMP5366 MLOps and Data Analytics Pipeline
