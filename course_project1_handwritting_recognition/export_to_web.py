"""
Standalone Weight Exporter: PyTorch checkpoint -> Web weights.js
Converts trained PyTorch model parameters into base64-encoded Float32 arrays for browser inference.
"""

import os
import json
import base64
import numpy as np
import torch
from train import DigitCNN


def export_checkpoint(model_path="./models/mnist_cnn.pt", output_path="./models/weights.js"):
    if not os.path.exists(model_path):
        print(f"[!] Error: {model_path} does not exist. Run train.py first.")
        return

    print(f"[*] Loading PyTorch model from {model_path}...")
    model = DigitCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    export_dict = {}
    for name, param in model.named_parameters():
        arr = param.detach().numpy().astype(np.float32)
        export_dict[name] = {
            "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii")
        }

    content = "/** Pre-trained MNIST CNN Weights (Trained with PyTorch) **/\n"
    content += "window.MODEL_WEIGHTS = " + json.dumps(export_dict) + ";\n"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"[✓] Successfully exported weights to {output_path} ({len(content):,} bytes).")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "./models/weights.js"
    export_checkpoint(output_path=target)

