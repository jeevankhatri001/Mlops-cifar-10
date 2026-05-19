# 🚀 CIFAR-10 MLOps Pipeline

An end-to-end MLOps pipeline for CIFAR-10 image classification using:

- PyTorch
- MLflow
- FastAPI
- PostgreSQL
- Redis
- Docker
- Apache Airflow
- AWS EC2

---

# 📌 Project Overview

This project demonstrates a complete MLOps workflow for:

- Training deep learning models
- Experiment tracking
- Workflow orchestration
- API deployment
- Containerization
- Cloud deployment

The system uses a CNN-based image classifier trained on the CIFAR-10 dataset.

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Dataset | CIFAR-10 |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Workflow Orchestration | Apache Airflow |
| Database | PostgreSQL |
| Message Broker | Redis |
| Containerization | Docker |
| Deployment | AWS EC2 Ubuntu |

---

# 📂 Project Structure

```bash
cifar10-mlops/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── training/
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   └── models/
│
├── airflow/
│   ├── dags/
│   │   └── cifar10_pipeline.py
│   │
│   └── docker-compose-airflow.yml
│
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ✨ Features

## ✅ Model Training
- CNN-based CIFAR-10 classifier
- Batch normalization
- Dropout regularization
- Data augmentation
- Adam optimizer

## ✅ Experiment Tracking
- MLflow integration
- Hyperparameter logging
- Metric tracking
- Model artifact storage

## ✅ API Deployment
- FastAPI REST API
- Image upload prediction endpoint
- Interactive Swagger UI

## ✅ Workflow Automation
- Apache Airflow DAG orchestration
- Automated training pipeline
- Dataset preparation
- Artifact verification

---

# 📊 Model Performance

| Metric | Result |
|---|---|
| Final Accuracy | ~77% |
| Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.001 |

---

# ☁️ AWS EC2 Setup

## Launch EC2 Instance

Recommended:
- Ubuntu 22.04
- t2.medium or higher

### Open Ports

| Port | Purpose |
|---|---|
| 22 | SSH |
| 5000 | MLflow |
| 8000 | FastAPI |
| 8080 | Airflow |

---

# 📥 Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-mlops.git

cd cifar10-mlops
```

---

# 🐍 Create Virtual Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🐳 Start PostgreSQL and Redis

## docker-compose.yml

```yaml
services:
  postgres:
    image: postgres:16
    container_name: cifar10_postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: mlops_db
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    container_name: cifar10_redis
    ports:
      - "6379:6379"
```

## Start Containers

```bash
docker compose up -d
```

---

# 📈 Run MLflow

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5000
```

Access MLflow:

```text
http://YOUR_EC2_IP:5000
```

---

# 🧠 Train Model

```bash
python -m app.training.train
```

Expected Output:

```text
Epoch [1/10] ...
Epoch [10/10] ...
Training completed!
```

---

# ⚡ Run FastAPI Server

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Access Swagger Docs:

```text
http://YOUR_EC2_IP:8000/docs
```

---

# 🔍 API Usage

## POST `/predict`

Upload an image and receive a prediction.

Supported classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

# 🌊 Airflow Setup

## Navigate to Airflow Directory

```bash
cd airflow
```

## Start Airflow Containers

```bash
docker compose -f docker-compose-airflow.yml up -d
```

Access Airflow:

```text
http://YOUR_EC2_IP:8080
```

Default credentials:

```text
Username: admin
Password: admin
```

---

# 🔄 Airflow DAG Pipeline

```text
check_environment
↓
prepare_dataset
↓
train_model
↓
verify_model_artifact
↓
pipeline_complete
```

---

# 📊 MLflow Experiment Tracking

Each training run logs:

- Accuracy
- Loss
- Hyperparameters
- Model artifacts

Experiment Name:

```text
cifar10_training
```

---

# 🐳 Docker Containers

| Container | Purpose |
|---|---|
| postgres | Database |
| redis | Message broker |
| airflow_scheduler | DAG scheduling |
| airflow_webserver | Airflow UI |
| airflow_worker | Task execution |

---

# 🧰 Useful Commands

## Check Running Containers

```bash
docker ps
```

## View Container Logs

```bash
docker logs CONTAINER_NAME
```

## Restart Airflow

```bash
docker compose -f docker-compose-airflow.yml restart
```

## Stop Containers

```bash
docker compose down
```

---

# ⚠️ Common Issues

## MLflow Connection Error

Ensure:

```python
mlflow.set_tracking_uri("http://host.docker.internal:5000")
```

---

## Permission Errors Inside Containers

Save models inside project directories:

```python
models/cifar10_cnn.pth
```

Avoid:

```text
/home/ubuntu
```

inside Docker containers.

---

# 🚀 Future Improvements

- CI/CD using GitHub Actions
- Kubernetes deployment
- Model monitoring
- GPU training
- Automated retraining
- Unit testing
- Terraform infrastructure

---

# 🎓 Learning Outcomes

This project demonstrates:

- End-to-end MLOps workflow
- Deep learning deployment
- Experiment tracking
- Workflow orchestration
- Cloud deployment
- Docker containerization

---

# 👨‍💻 Author

**Jeevan Khatri**

BSc Computer Science with AI  
MLOps | Computer Vision | AI Engineering
