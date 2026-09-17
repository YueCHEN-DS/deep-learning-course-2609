"""Standalone Weight Exporter for Hotel Review MLP.

Converts trained PyTorch model parameters from best_model.pt into
base64-encoded Float32 arrays and JSON format for web transparency
and browser-side inspection (matching course_project1_handwritting_recognition).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL4_PATH = ROOT / "experiments" / "run_model4_champion_2000v4" / "best_model.pt"
MODEL3_PATH = ROOT / "experiments" / "run_model3_champion_1000v3" / "best_model.pt"
MODEL2_PATH = ROOT / "experiments" / "run_model2_optimized_500data" / "best_model.pt"
MODEL1_PATH = ROOT / "experiments" / "run_B_default_128_64" / "best_model.pt"

if MODEL4_PATH.exists():
    MODEL_PATH = MODEL4_PATH
elif MODEL3_PATH.exists():
    MODEL_PATH = MODEL3_PATH
elif MODEL2_PATH.exists():
    MODEL_PATH = MODEL2_PATH
else:
    MODEL_PATH = MODEL1_PATH

OUTPUT_JS = ROOT / "mlp_weights.js"
OUTPUT_JSON = ROOT / "mlp_weights.json"


def export_weights(
    model_path: Path = MODEL_PATH,
    out_js: Path = OUTPUT_JS,
    out_json: Path | None = None,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    print(f"[*] Loading PyTorch checkpoint from {model_path}...")
    chk = torch.load(model_path, map_location="cpu", weights_only=False)

    state_dict = chk["model_state_dict"]
    model_config = chk.get("model_config", {})

    export_dict = {}
    for name, param in state_dict.items():
        arr = param.detach().numpy().astype(np.float32)
        export_dict[name] = {
            "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        }

    # 1. Export to JS (window.MODEL_WEIGHTS)
    js_content = "/** Pre-trained Hotel Review MLP Weights (Trained with PyTorch) **/\n"
    js_content += f"window.MODEL_CONFIG = {json.dumps(model_config, ensure_ascii=False, indent=2)};\n\n"
    js_content += f"window.MODEL_WEIGHTS = {json.dumps(export_dict)};\n"

    out_js.write_text(js_content, encoding="utf-8")
    print(f"[✓] Successfully exported JS weights to: {out_js.name} ({len(js_content):,} bytes)")

    # 2. Export to JSON (optional)
    if out_json is not None:
        json_weights = {
            name: {"shape": list(param.shape), "values": param.detach().numpy().astype(np.float32).tolist()}
            for name, param in state_dict.items()
        }
        full_export = {
            "model_config": model_config,
            "weights": json_weights,
            "best_epoch": chk.get("best_epoch"),
            "val_metrics": chk.get("val_metrics"),
        }
        out_json.write_text(json.dumps(full_export, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[✓] Successfully exported JSON weights to: {out_json.name}")


if __name__ == "__main__":
    export_weights()
