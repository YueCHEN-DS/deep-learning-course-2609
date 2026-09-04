"""
Classical Machine Learning Model for MNIST Digit Recognition (Scikit-Learn)
Trains a non-neural network model: Multinomial Logistic Regression (Softmax Linear Classifier).
Evaluates performance, saves confusion matrix, and exports weights for browser inference.
"""

import os
import json
import base64
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def train_sklearn():
    print("[*] Loading MNIST dataset for Scikit-Learn training...")
    train_ds = datasets.MNIST(root="./data", train=True, download=False, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root="./data", train=False, download=False, transform=transforms.ToTensor())

    # Flatten 28x28 images to 784-dimensional feature vectors
    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy()

    X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy()

    print(f"[*] Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"[*] Testing samples:  {X_test.shape[0]}")

    # Initialize Multinomial Logistic Regression (L-BFGS solver)
    print("\n[*] Training Multinomial Logistic Regression (Softmax Classifier)...")
    start_time = time.time()
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=250,
        C=1.0,
        random_state=42
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"[✓] Scikit-Learn training completed in {elapsed:.2f}s!")

    # Evaluate on test set
    test_acc = clf.score(X_test, y_test) * 100.0
    print(f"[*] Test Accuracy: {test_acc:.2f}%")

    y_pred = clf.predict(X_test)

    # 1. Classification Report
    target_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n" + "="*50)
    print("   SCIKIT-LEARN CLASSIFICATION REPORT (MNIST)")
    print("="*50)
    print(report)

    with open("./models/sklearn_report.txt", "w") as f:
        f.write(report)

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs("./plots", exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f"Confusion Matrix - Scikit-Learn Logistic Regression ({test_acc:.2f}%)", fontsize=13, fontweight='bold', pad=15)
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
    print("[✓] Saved confusion matrix to ./plots/sklearn_confusion_matrix.png")

    # 3. Save Scikit-Learn Joblib Checkpoint
    os.makedirs("./models", exist_ok=True)
    joblib.dump(clf, "./models/sklearn_logistic_regression.joblib")
    print("[*] Saved joblib model to ./models/sklearn_logistic_regression.joblib")

    # 4. Export Weights for Web Application
    coef = clf.coef_.astype(np.float32)         # Shape: (10, 784)
    intercept = clf.intercept_.astype(np.float32) # Shape: (10,)

    export_dict = {
        "model_name": "Logistic Regression (Scikit-Learn)",
        "accuracy": round(test_acc, 2),
        "coef": {
            "shape": list(coef.shape),
            "b64": base64.b64encode(coef.tobytes()).decode("ascii")
        },
        "intercept": {
            "shape": list(intercept.shape),
            "b64": base64.b64encode(intercept.tobytes()).decode("ascii")
        }
    }

    content = "/** Scikit-Learn Logistic Regression Weights **/\n"
    content += "window.SKLEARN_WEIGHTS = " + json.dumps(export_dict) + ";\n"

    local_web_path = "./sklearn_weights.js"
    with open(local_web_path, "w") as f:
        f.write(content)
    print(f"[✓] Saved weights directly to {local_web_path}")


if __name__ == "__main__":
    train_sklearn()
