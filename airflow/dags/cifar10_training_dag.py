from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"owner": "jeevan"}

PROJECT = "/opt/airflow/project"

# The pipeline code runs INSIDE the Airflow containers, which sit on a different
# Docker network from the app's Postgres/Redis/MLflow. So the tasks must reach
# those services on the HOST via host.docker.internal (mapped in
# docker-compose-airflow.yml) instead of 'localhost'.
STORE_ENV = (
    "MARIADB_HOST=host.docker.internal "
    "MARIADB_PORT=3306 "
    "MARIADB_USER=mlops "
    "MARIADB_PASSWORD=mlops123 "
    "MARIADB_DB=mlops_db "
    "REDIS_HOST=host.docker.internal "
    "MLFLOW_TRACKING_URI=http://host.docker.internal:5000"
)

with DAG(
    dag_id="cifar10_training_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "cifar10", "training"],
) as dag:

    check_environment = BashOperator(
        task_id="check_environment",
        bash_command=f"""
        cd {PROJECT} && \
        python -c "import torch, mlflow; print('Environment OK')" && \
        {STORE_ENV} python -c "from app.database.db import engine; from sqlalchemy import text; c=engine.connect(); c.execute(text('SELECT 1')); c.close(); print('Store reachable')"
        """,
    )

    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=f"""
        cd {PROJECT} && \
        {STORE_ENV} python -m app.ingestion.ingest
        """,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"""
        cd {PROJECT} && \
        {STORE_ENV} EPOCHS=10 python -m app.training.train
        """,
    )

    verify_model_artifact = BashOperator(
        task_id="verify_model_artifact",
        bash_command=f"""
        cd {PROJECT} && \
        test -f models/cifar10_cnn.pth && echo 'Model artifact exists'
        """,
    )

    pipeline_complete = BashOperator(
        task_id="pipeline_complete",
        bash_command="echo 'CIFAR-10 MLOps training pipeline completed successfully'",
    )

    check_environment >> ingest_data >> train_model >> verify_model_artifact >> pipeline_complete
