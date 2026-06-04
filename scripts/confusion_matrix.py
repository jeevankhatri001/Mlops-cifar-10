"""
Generate a confusion matrix from the trained CIFAR-10 model.
Run from the repo root on your EC2 box:

    cd ~/cifar10-mlops
    source venv/bin/activate
    python -m scripts.confusion_matrix

Output: confusion_matrix.png in the repo root.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from app.training.model import CIFAR10CNN
from app.training.dataset import get_dataloaders_from_store

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MODEL_PATH = os.getenv("MODEL_PATH", "models/cifar10_cnn.pth")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Loading model from {MODEL_PATH}")

    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders_from_store(batch_size=64)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    print(f"Test accuracy: {acc:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"CIFAR-10 Confusion Matrix (test accuracy {acc:.1f}%)",
                 fontsize=13, fontweight="bold", pad=12)

    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#333",
                    fontsize=8)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
    print("\nSaved confusion_matrix.png")


if __name__ == "__main__":
    main()
