"""
Evaluation and Metrics Generation for MNIST CNN
Generates confusion matrix, precision/recall report, and training curves.
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
from train import DigitCNN, get_device


def evaluate():
    device = get_device()
    print(f"[*] Loading model onto {device}...")

    model = DigitCNN().to(device)
    model_path = "./models/mnist_cnn.pt"
    if not os.path.exists(model_path):
        print(f"[!] Model file {model_path} not found. Please run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
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

    # 1. Classification Report
    target_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    print("\n" + "="*50)
    print("      CLASSIFICATION REPORT (MNIST Test Set)")
    print("="*50)
    print(report)

    with open("./models/classification_report.txt", "w") as f:
        f.write(report)

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    os.makedirs("./plots", exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MNIST Digit Recognition", fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)])
    plt.yticks(tick_marks, [str(i) for i in range(10)])

    # Annotate cells
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
    plt.savefig("./plots/confusion_matrix.png", dpi=300)
    plt.close()
    print("[✓] Saved confusion matrix plot to ./plots/confusion_matrix.png")

    # 3. Training & Validation Curves (if history exists)
    history_file = "./models/training_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Loss Curve
        ax1.plot(history["epochs"], history["train_loss"], 'o-', color='#2563eb', label='Train Loss', linewidth=2)
        ax1.plot(history["epochs"], history["test_loss"], 's--', color='#f43f5e', label='Test Loss', linewidth=2)
        ax1.set_title("Cross-Entropy Loss vs. Epoch", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # Accuracy Curve
        ax2.plot(history["epochs"], history["test_accuracy"], 'o-', color='#10b981', label='Test Accuracy (%)', linewidth=2)
        ax2.set_title("Test Accuracy vs. Epoch", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim([95, 100])
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("./plots/training_curves.png", dpi=300)
        plt.close()
        print("[✓] Saved training curves to ./plots/training_curves.png")

    # 4. Sample Test Predictions Plot
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()
    for idx in range(10):
        img, true_lbl = test_ds[idx]
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_lbl = prob.argmax()
            conf = prob[pred_lbl] * 100

        axes[idx].imshow(img.squeeze(), cmap='gray')
        color = 'green' if pred_lbl == true_lbl else 'red'
        axes[idx].set_title(f"Pred: {pred_lbl} ({conf:.1f}%)\nTrue: {true_lbl}", color=color, fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle("Sample Test Predictions (PyTorch CNN)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./plots/sample_predictions.png", dpi=300)
    plt.close()
    print("[✓] Saved sample predictions plot to ./plots/sample_predictions.png")


if __name__ == "__main__":
    evaluate()
