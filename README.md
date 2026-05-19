CIFAR-10 MLOps Pipeline

An end-to-end MLOps pipeline for CIFAR-10 image classification using PyTorch, MLflow, FastAPI, PostgreSQL, Redis, Docker, and Apache Airflow.

Project Overview

This project demonstrates a complete MLOps workflow for training, tracking, orchestrating, and serving a deep learning image classification model.

The system includes:

CNN model training using PyTorch
Experiment tracking using MLflow
API deployment using FastAPI
Workflow orchestration using Apache Airflow
Redis and PostgreSQL integration
Dockerized infrastructure
AWS EC2 deployment
Tech Stack
Component	Technology
Deep Learning	PyTorch
Dataset	CIFAR-10
Experiment Tracking	MLflow
API Framework	FastAPI
Workflow Orchestration	Apache Airflow
Database	PostgreSQL
Message Broker	Redis
Containerization	Docker
Deployment	AWS EC2 Ubuntu
Project Structure
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
Features
Model Training
CNN-based CIFAR-10 classifier
Data augmentation
Batch normalization
Dropout regularization
Adam optimizer
Experiment Tracking
MLflow integration
Metric logging
Model artifact storage
Parameter tracking
API Deployment
FastAPI REST API
Image upload prediction endpoint
Swagger documentation
Workflow Automation
Airflow DAG orchestration
Automated training pipeline
Dataset preparation
Artifact verification
Model Performance
Metric	Result
Final Accuracy	~77%
Epochs	10
Batch Size	64
Learning Rate	0.001
AWS EC2 Setup
Launch EC2 Instance

Recommended:

Ubuntu 22.04
t2.medium or higher

Open ports:

22 (SSH)
5000 (MLflow)
8000 (FastAPI)
8080 (Airflow)
Clone Repository
git clone https://github.com/YOUR_USERNAME/cifar10-mlops.git

cd cifar10-mlops
Create Virtual Environment
python3 -m venv venv

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Start PostgreSQL and Redis
docker-compose.yml
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
Run Containers
docker compose up -d
Run MLflow
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5000

Access:

http://YOUR_EC2_IP:5000
Train Model
python -m app.training.train

Expected output:

Epoch [1/10] ...
Epoch [10/10] ...
Training completed!
Run FastAPI Server
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

Access Swagger Docs:

http://YOUR_EC2_IP:8000/docs
API Usage
Prediction Endpoint
POST /predict

Upload an image and receive prediction.

Supported classes:

airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
Airflow Setup
Navigate to Airflow Folder
cd airflow
Start Airflow Containers
docker compose -f docker-compose-airflow.yml up -d
Access Airflow
http://YOUR_EC2_IP:8080

Default credentials:

Username: admin
Password: admin
Airflow DAG Pipeline

The Airflow pipeline includes:

check_environment
↓
prepare_dataset
↓
train_model
↓
verify_model_artifact
↓
pipeline_complete
Run DAG
Open Airflow UI
Enable DAG
Trigger DAG manually
MLflow Experiment Tracking

Each run logs:

Training accuracy
Loss
Hyperparameters
Model artifacts

Experiment name:

cifar10_training
Docker Containers Used
Container	Purpose
postgres	Database
redis	Message broker
airflow_scheduler	Workflow scheduling
airflow_webserver	Airflow UI
airflow_worker	Task execution
Important Commands
Check Running Containers
docker ps
View Logs
docker logs CONTAINER_NAME
Restart Airflow
docker compose -f docker-compose-airflow.yml restart
Stop Containers
docker compose down
Common Issues
Airflow Logs Not Loading

Fix:

Ensure ports are open
Restart Airflow services
MLflow Connection Error

Ensure:

mlflow.set_tracking_uri("http://host.docker.internal:5000")
Permission Errors

Use writable project directories:

models/cifar10_cnn.pth

Avoid saving to:

/home/ubuntu

inside containers.

Future Improvements
CI/CD pipeline using GitHub Actions
Kubernetes deployment
Model monitoring
GPU training
Automated retraining
Unit testing
Terraform infrastructure
Learning Outcomes

This project demonstrates:

End-to-end MLOps workflow
Model orchestration
Experiment tracking
API deployment
Docker containerization
Cloud deployment
Workflow automation
Author

Jeevan Khatri

BSc Computer Science with AI
MLOps | Computer Vision | AI Engineering
