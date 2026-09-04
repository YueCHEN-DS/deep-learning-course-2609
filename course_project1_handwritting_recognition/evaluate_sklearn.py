"""
Evaluation Script for Scikit-Learn Model (Classical ML)
Loads saved joblib model, computes metrics on test set, and updates confusion matrix.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def evaluate_sklearn():
    model_path = "./models/sklearn_logistic_regression.joblib"
    if not os.path.exists(model_path):
        print(f"[!] Model {model_path} not found. Run train_sklearn.py first.")
        return

    print(f"[*] Loading Scikit-Learn model from {model_path}...")
    clf = joblib.load(model_path)

    test_ds = datasets.MNIST(root="./data", train=False, download=False, transform=transforms.ToTensor())
    X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy()

    test_acc = clf.score(X_test, y_test) * 100.0
    y_pred = clf.predict(X_test)

    print("\n" + "="*50)
    print(f"   SCIKIT-LEARN EVALUATION REPORT (Test Acc: {test_acc:.2f}%)")
    print("="*50)
    target_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs("./plots", exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f"Confusion Matrix - Scikit-Learn ML ({test_acc:.2f}%)", fontsize=13, fontweight='bold', pad=15)
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
    plt.savefig("./plots/sklearn_confusion_matrix.png", dpi=300)
    plt.close()
    print("[✓] Updated plot: ./plots/sklearn_confusion_matrix.png")


if __name__ == "__main__":
    evaluate_sklearn()
