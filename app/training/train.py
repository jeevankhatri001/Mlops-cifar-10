import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sqlalchemy import text

from app.training.dataset import get_dataloaders_from_store
from app.training.model import CIFAR10CNN
from app.training.evaluate import evaluate_model
from app.database.db import engine, init_db

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/cifar10_cnn.pth")

# Use the MLflow server if configured, else a local file store (works offline)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "file:./mlruns"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("cifar10_training")


def log_experiment_to_db(run_id, accuracy, loss):
    with engine.begin() as conn:
        conn.execute(text(
            """
            INSERT INTO experiments
                (experiment_name, run_id, accuracy, loss,
                 epochs, learning_rate, batch_size)
            VALUES (:name, :run_id, :acc, :loss, :epochs, :lr, :bs)
            """
        ), {"name": "cifar10_training", "run_id": run_id, "acc": accuracy,
            "loss": loss, "epochs": EPOCHS, "lr": LEARNING_RATE, "bs": BATCH_SIZE})


def main():
    init_db()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | MLflow: {TRACKING_URI} | epochs: {EPOCHS}")

    train_loader, test_loader = get_dataloaders_from_store(batch_size=BATCH_SIZE)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS, "data_source": "analytics_store_postgres_redis",
            "model_type": "cnn_batchnorm_dropout",
        })

        final_loss = final_acc = 0.0
        for epoch in range(EPOCHS):
            model.train()
            running = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running += loss.item()
            final_loss = running / len(train_loader)
            final_acc = evaluate_model(model, test_loader, device)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {final_loss:.4f} Acc: {final_acc:.2f}%")
            mlflow.log_metric("loss", final_loss, step=epoch)
            mlflow.log_metric("accuracy", final_acc, step=epoch)

        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        try:
            mlflow.log_artifact(MODEL_PATH)
        except Exception as e:
            print(f"(warning) MLflow artifact upload skipped: {e}")
        log_experiment_to_db(run.info.run_id, final_acc, final_loss)

        print(f"Done. Model -> {MODEL_PATH}")
        print(f"Logged run {run.info.run_id} to experiments table + MLflow.")


if __name__ == "__main__":
    main()
