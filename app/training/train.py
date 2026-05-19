import os

import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim

from app.training.dataset import get_dataloaders
from app.training.model import CIFAR10CNN
from app.training.evaluate import evaluate_model


mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment("cifar10_training")


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10


train_loader, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
)

model = CIFAR10CNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


with mlflow.start_run():

    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("model_type", "improved_cnn_batchnorm_dropout")
    mlflow.log_param("augmentation", "random_crop_horizontal_flip")
    

    for epoch in range(EPOCHS):

        running_loss = 0.0

        model.train()

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        accuracy = evaluate_model(
            model,
            test_loader
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

        mlflow.log_metric(
            "loss",
            epoch_loss,
            step=epoch
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
            step=epoch
        )

    os.makedirs("models", exist_ok=True)

    model_path = "models/cifar10_cnn.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

   # mlflow.log_artifact(model_path)

  #  mlflow.pytorch.log_model(
   #     model,
 #       "cifar10_model"
#    )

    print("Training completed!")
