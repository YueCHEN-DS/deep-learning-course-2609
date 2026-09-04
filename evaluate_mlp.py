"""
Evaluation and Metrics Generation for MNIST MLP
Generates confusion matrix, precision/recall report, and training curves for MLP.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from train_mlp import DigitMLP, get_device


def evaluate_mlp():
    device = get_device()
    print(f"[*] Loading MLP model onto {device}...")

    model = DigitMLP().to(device)
    model_path = "./models/mnist_mlp.pt"
    if not os.path.exists(model_path):
        print(f"[!] Model file {model_path} not found. Please run train_mlp.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = datasets.MNIST(root="./data", train=False, download=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500, shuffle=False)

    all_preds = []
    all_targets = []

    print("[*] Running evaluation on 10,000 test images...")
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Classification Report
    target_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    print("\n" + "="*50)
    print("      CLASSIFICATION REPORT (MNIST MLP)")
    print("="*50)
    print(report)

    with open("./models/mlp_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    os.makedirs("./plots", exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title("Confusion Matrix - PyTorch MLP", fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)])
    plt.yticks(tick_marks, [str(i) for i in range(10)])

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=9)

    plt.ylabel('True Digit', fontsize=12, fontweight='600')
    plt.xlabel('Predicted Digit', fontsize=12, fontweight='600')
    plt.tight_layout()
    plt.savefig("./plots/mlp_confusion_matrix.png", dpi=300)
    plt.close()
    print("[✓] Saved confusion matrix plot to ./plots/mlp_confusion_matrix.png")

    # Training Curves
    history_file = "./models/mlp_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(history["epochs"], history["train_loss"], 'o-', color='#7c3aed', label='Train Loss', linewidth=2)
        ax1.plot(history["epochs"], history["test_loss"], 's--', color='#f43f5e', label='Test Loss', linewidth=2)
        ax1.set_title("MLP Cross-Entropy Loss vs. Epoch", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        ax2.plot(history["epochs"], history["test_accuracy"], 'o-', color='#7c3aed', label='Test Accuracy (%)', linewidth=2)
        ax2.set_title("MLP Test Accuracy vs. Epoch", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim([90, 100])
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("./plots/mlp_training_curves.png", dpi=300)
        plt.close()
        print("[✓] Saved training curves to ./plots/mlp_training_curves.png")


if __name__ == "__main__":
    evaluate_mlp()
