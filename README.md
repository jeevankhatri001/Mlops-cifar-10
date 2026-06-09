cat > ~/cifar10-mlops/README.md << 'EOF'
# CIFAR-10 MLOps Pipeline

An end-to-end MLOps pipeline for CIFAR-10 image classification, built as a CMP5366 coursework project. The system implements a complete data-analytics and machine-learning pipeline with polyglot persistence (MariaDB Columnstore + Redis), automated orchestration, live inference with drift monitoring, and exploratory data analysis.

## Tech Stack

| Component | Technology | Role in Pipeline |
|---|---|---|
| Deep Learning | PyTorch | CNN model training and inference |
| Dataset | CIFAR-10 (60,000 images) | Supervised image classification |
| Relational Store | MariaDB 10.11 (Columnstore) | Analytics store — image features, experiments, predictions (OBT design) |
| Key-Value Store | Redis 7 | Fast pixel-array storage and retrieval for training/inference |
| Experiment Tracking | MLflow | Hyperparameter, metric, and artifact logging |
| API Framework | FastAPI | REST inference endpoint with Swagger UI |
| Frontend | Streamlit | Interactive dashboard — Predict, Monitoring, Drift Detection |
| Monitoring | Custom (z-score) + Evidently AI | Per-request and population-level data-drift detection |
| Orchestration | Apache Airflow | DAG-based pipeline automation |
| Containerisation | Docker Compose | Service management for Redis and Airflow |
| Deployment | AWS EC2 (Ubuntu 24.04) | Cloud hosting |

## Project Structure
cifar10-mlops/
├── app/
│   ├── api/
│   │   ├── init.py
│   │   └── main.py                # FastAPI app — /predict, /monitoring/stats, /monitoring/drift
│   ├── database/
│   │   ├── db.py                  # SQLAlchemy engine (mysql+pymysql) + init_db()
│   │   ├── redis_client.py        # Redis connection pool
│   │   └── schema.sql             # Analytics store schema (MariaDB Columnstore)
│   ├── inference/
│   │   ├── init.py
│   │   └── predict.py             # Lazy model loading, prediction + feature extraction
│   ├── ingestion/
│   │   ├── init.py
│   │   ├── features.py            # Per-image feature engineering (brightness, contrast, RGB means)
│   │   └── ingest.py              # ETL: torchvision → MariaDB (features) + Redis (pixels)
│   ├── monitoring/
│   │   ├── init.py
│   │   ├── baseline.py            # Training-set feature distribution (drift reference)
│   │   ├── drift.py               # Z-score drift detection
│   │   ├── evidently_monitor.py   # Evidently AI population drift report
│   │   └── monitor.py             # Prediction logging + population drift report
│   └── training/
│       ├── dataset.py             # DataLoaders — store-backed (MariaDB labels + Redis pixels)
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
│   └── eda.ipynb                  # EDA — connects to MariaDB analytics store
├── scripts/
│   └── export_sample.py           # Pull real CIFAR images from Redis → PNG for testing
├── streamlit_app.py               # Streamlit dashboard (port 8501)
├── docker-compose.yml             # Redis for the app
├── requirements.txt
├── .env                           # Credentials (gitignored)
├── .gitignore
└── README.md
## Architecture

### Storage Design (Polyglot Persistence)

The pipeline uses a **denormalised One Big Table (OBT)** in MariaDB Columnstore for structured, queryable metadata and a **Redis key-value store** for raw pixel arrays:

| Store | Table / Key Pattern | Purpose |
|---|---|---|
| MariaDB Columnstore | `images` | Per-image feature columns (brightness, contrast, RGB means/stds, label, split) for EDA and drift baseline |
| MariaDB Columnstore | `experiments` | Training run log (accuracy, loss, hyperparams, MLflow run ID) |
| MariaDB Columnstore | `predictions` | Live inference log (predicted class, confidence, latency, input features, drift flag) |
| Redis | `img:{split}:{idx}` | Raw 32×32×3 uint8 pixel arrays for fast training/inference retrieval |

> **Note:** MariaDB Columnstore plugin is installed and active on the host. Tables use `ENGINE=Columnstore` in the schema definition. Due to a known Columnstore worker stability issue on Ubuntu 24.04 with single-node deployments, tables were migrated to InnoDB engine while retaining identical columnar-style analytics queries (`avg`, `stddev_samp`, `GROUP BY`).

### Pipeline Stages
Data Ingestion     torchvision → feature extraction → MariaDB + Redis
Data Preprocessing normalisation, augmentation (transforms in DataLoader)
Model Development  CNN training from the store, logged to MLflow + experiments table
Model Deployment   serialised .pth, served via FastAPI /predict
Model Monitoring   every prediction logged; z-score drift detection per request + population level
Frontend           Streamlit dashboard — Predict / Monitoring / Drift Detection tabs

### Airflow DAG
check_environment → ingest_data → train_model → verify_model_artifact → pipeline_complete
Tasks run inside the Airflow worker container and reach the app's MariaDB/Redis/MLflow on the host via `host.docker.internal`.

## Model Performance

| Metric | Value |
|---|---|
| Architecture | 3-block CNN (BatchNorm + Dropout) |
| Test Accuracy | ~78% (10 epochs) |
| Optimiser | Adam (lr=0.001) |
| Batch Size | 64 |
| Data Source | Analytics store (MariaDB labels + Redis pixels) |

## Setup

### Prerequisites

- AWS EC2 Ubuntu 24.04 instance (t2.medium or higher)
- Docker and Docker Compose installed
- Python 3.9+ with venv
- Security group ports open: 22 (SSH), 3306 (MariaDB), 5000 (MLflow), 8000 (FastAPI), 8501 (Streamlit), 8080 (Airflow)

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/jeevankhatri001/Mlops-cifar-10.git
cd cifar10-mlops
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pymysql cryptography
```

### 2. Install MariaDB with Columnstore

```bash
# Download official MariaDB repo setup
curl -LsS https://r.mariadb.com/downloads/mariadb_repo_setup -o mariadb_repo_setup.sh
sudo bash mariadb_repo_setup.sh --mariadb-server-version="mariadb-10.11"

# Install MariaDB + Columnstore plugin
sudo apt update
sudo apt install -y mariadb-server mariadb-plugin-columnstore libmariadb-dev libmariadb-dev-compat

# Start MariaDB
sudo systemctl start mariadb
sudo systemctl enable mariadb
```

### 3. Configure MariaDB

```bash
sudo mariadb -u root -p
```

```sql
INSTALL SONAME 'ha_columnstore';
CREATE DATABASE mlops_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'mlops'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON mlops_db.* TO 'mlops'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 4. Configure Environment Variables

Create `.env` in the repo root (gitignored — never commit this):

```env
MARIADB_HOST=localhost
MARIADB_PORT=3306
MARIADB_USER=mlops
MARIADB_PASSWORD=your_password
MARIADB_DB=mlops_db

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

MLFLOW_TRACKING_URI=http://localhost:5000
```

### 5. Start Redis

```bash
docker compose up -d redis
```

### 6. Initialise the Database Schema

```bash
python -m app.database.db
# Expected: Tables in database: ['experiments', 'images', 'predictions']
```

### 7. Ingest CIFAR-10 into the Analytics Store

```bash
python -m app.ingestion.ingest
# Ingests 50,000 train + 10,000 test images (features → MariaDB, pixels → Redis)
```

### 8. Start MLflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000 &
```

### 9. Train the Model

```bash
python -m app.training.train
# Reads from the store, logs to MLflow + experiments table, saves models/cifar10_cnn.pth
```

### 10. Run the API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### 11. Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root check |
| GET | `/health` | Health check |
| POST | `/predict` | Upload an image → prediction + confidence + drift check |
| GET | `/monitoring/stats` | Total predictions, drift count, class distribution |
| GET | `/monitoring/drift?window=100` | Population drift report vs training baseline |

Swagger UI: `http://<EC2_IP>:8000/docs`

## Monitoring and Drift Detection

Every `/predict` call:

1. Computes image features (brightness, contrast, RGB channel means) on the resized 32×32 input
2. Runs a **z-score drift check** against the training baseline (flags if any feature > 3σ from the mean)
3. Logs the prediction, features, confidence, latency, and drift flag to the `predictions` table in MariaDB

The `/monitoring/drift` endpoint aggregates the last N predictions and flags **population-level drift** when the recent feature means shift more than 2σ from baseline.

Evidently AI drift report can be generated separately:

```bash
python -m app.monitoring.evidently_monitor
# Generates: evidently_drift_report.html
```

## Useful Commands

```bash
# Check MariaDB tables
mariadb -u mlops -pmlops123 mlops_db -e "SHOW TABLES;"

# Verify ingested data
mariadb -u mlops -pmlops123 mlops_db -e "SELECT split, count(*) AS total FROM images GROUP BY split;"

# Check experiment history
mariadb -u mlops -pmlops123 mlops_db -e "SELECT id, run_id, accuracy, epochs, created_at FROM experiments ORDER BY id DESC LIMIT 5;"

# Check prediction log
mariadb -u mlops -pmlops123 mlops_db -e "SELECT id, predicted_class, confidence, drift_flag, requested_at FROM predictions ORDER BY id DESC LIMIT 10;"

# Redis key count
docker exec -it cifar10_redis redis-cli DBSIZE

# Restart MariaDB Columnstore workers
sudo systemctl stop mariadb
sudo systemctl stop mariadb-columnstore
sudo systemctl start mariadb-columnstore
sudo systemctl start mariadb
```

## CIFAR-10 Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Author

**Jeevan Khatri**
Birmingham City University — CMP5366 MLOps and Data Analytics Pipeline
EOF
