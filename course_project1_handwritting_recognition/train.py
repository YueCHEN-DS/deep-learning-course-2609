"""
Handwritten Digit Recognition - PyTorch Training Pipeline
Trained on MNIST with data augmentation for robust real-world drawing recognition.
"""

import os
import json
import base64
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class DigitCNN(nn.Module):
    """
    Compact 2-stage CNN for MNIST digit classification (~97k parameters).
    Architecture:
      Conv1 (1 -> 16, 3x3, pad 1) -> ReLU -> MaxPool (2x2) [16 x 14 x 14]
      Conv2 (16 -> 32, 3x3, pad 1) -> ReLU -> MaxPool (2x2) [32 x 7 x 7 = 1568]
      FC1 (1568 -> 96) -> ReLU -> Dropout(0.2)
      FC2 (96 -> 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 96)
        self.fc2 = nn.Linear(96, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_device():
    """Detect Apple Silicon GPU (MPS) or fallback to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def export_weights_to_js(model, output_js_path):
    """Exports PyTorch weights directly to a JavaScript file for client-side inference."""
    export_dict = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy().astype(np.float32)
        export_dict[name] = {
            "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii")
        }

    header = "/** Pre-trained MNIST CNN Weights (Trained with PyTorch) **/\n"
    content = header + "window.MODEL_WEIGHTS = " + json.dumps(export_dict) + ";\n"
    with open(output_js_path, "w") as f:
        f.write(content)
    print(f"[*] Exported model weights to web app: {output_js_path}")


def train(epochs=5, batch_size=128, lr=0.002, data_dir="./data"):
    device = get_device()
    print(f"[*] Using compute device: {device}")

    # Data augmentation for robust real-world drawing recognition
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()

    print("[*] Loading MNIST dataset...")
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500, shuffle=False)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": []
    }

    print(f"\n[*] Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        total_train = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            total_train += data.size(0)

        epoch_train_loss = running_loss / total_train

        # --- Evaluation phase ---
        model.eval()
        running_test_loss = 0.0
        correct = 0
        total_test = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)

                running_test_loss += loss.item() * data.size(0)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                total_test += data.size(0)

        epoch_test_loss = running_test_loss / total_test
        epoch_test_acc = (correct / total_test) * 100.0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["test_loss"].append(round(epoch_test_loss, 4))
        history["test_accuracy"].append(round(epoch_test_acc, 2))

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Test Loss: {epoch_test_loss:.4f} | "
              f"Test Accuracy: {epoch_test_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n[✓] Training completed in {elapsed:.1f}s! Final Accuracy: {history['test_accuracy'][-1]}%")

    # Save PyTorch checkpoint
    os.makedirs("./models", exist_ok=True)
    model_path = "./models/mnist_cnn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[*] Saved PyTorch model to {model_path}")

    # Save training history
    with open("./models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save exported JS weights in the same folder for index.html
    local_weights_path = "./weights.js"
    export_weights_to_js(model, local_weights_path)

    return model, history


if __name__ == "__main__":
    train(epochs=5)
