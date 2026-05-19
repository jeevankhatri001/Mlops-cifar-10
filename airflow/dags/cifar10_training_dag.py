from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "jeevan",
}


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
        bash_command="""
        cd /opt/airflow/project && \
        python --version && \
        python -c "import torch, mlflow; print('Environment OK')"
        """
    )

    prepare_dataset = BashOperator(
        task_id="prepare_dataset",
        bash_command="""
        cd /opt/airflow/project && \
        python -c "from app.training.dataset import get_dataloaders; get_dataloaders(batch_size=64); print('Dataset ready')"
        """
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
        cd /opt/airflow/project && \
        python -m app.training.train
        """
    )

    verify_model_artifact = BashOperator(
        task_id="verify_model_artifact",
        bash_command="""
        cd /opt/airflow/project && \
        test -f models/cifar10_cnn.pth && echo 'Model artifact exists'
        """
    )

    pipeline_complete = BashOperator(
        task_id="pipeline_complete",
        bash_command="echo 'CIFAR-10 MLOps training pipeline completed successfully'"
    )

    check_environment >> prepare_dataset >> train_model >> verify_model_artifact >> pipeline_complete
