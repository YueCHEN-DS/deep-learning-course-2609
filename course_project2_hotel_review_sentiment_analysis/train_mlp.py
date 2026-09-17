"""MLP Training Script for Hotel Review Aspect-Sentiment Analysis.

Designed strictly according to the course project requirements:
- Slides 24-26: MLP Architecture (256 -> hidden_layers -> 8 aspects × 5 states = 40 logits)
- Slide 22: Group train/val/test splitting by review_id to prevent data leakage
- Slide 26: CrossEntropyLoss on logits.reshape(-1, 5) vs y.reshape(-1)
- Slide 27: Comprehensive evaluation (Mention Detection P/R/F1, End-to-end Macro/Micro F1,
            Per-aspect breakdowns, Class-imbalance handling, and Top 5 error analysis)
- Slide 25: Reproducible experiment logging (seed, hidden widths, params, lr, dropout)

Usage:
    # Run B (Default architecture: 128 -> 64)
    python train_mlp.py --labels-file labels_ai_suggestions.jsonl --allow-unreviewed

    # Run A (Compact model: 64)
    python train_mlp.py --hidden-dims 64 --exp-name run_A_64

    # Run C (Deeper model: 128 -> 64 -> 32)
    python train_mlp.py --hidden-dims 128 64 32 --exp-name run_C_128_64_32
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Matplotlib for loss curve plotting (optional)
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Scikit-learn for evaluation metrics
try:
    from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ==============================================================================
# 1. Project Constants (Strictly aligned with Slides 7 & 17)
# ==============================================================================

ASPECTS = [
    "cleanliness",              # 0
    "service",                  # 1
    "location",                 # 2
    "facilities",               # 3
    "room_comfort",             # 4
    "sound_insulation_noise",   # 5
    "food",                     # 6
    "value",                    # 7
]

STATES = {
    0: "absent",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "mixed",
}

NUM_ASPECTS = len(ASPECTS)  # 8
NUM_STATES = len(STATES)    # 5
INPUT_DIM = 256             # Qwen 256-dimensional embedding


# ==============================================================================
# 2. PyTorch Dataset Definition
# ==============================================================================

class HotelAspectDataset(Dataset):
    """PyTorch Dataset holding 256-dim embeddings and 8-dim state target vectors."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sentence_ids: list[str],
        sentences: list[str],
        review_ids: list[str],
    ):
        assert len(embeddings) == len(labels) == len(sentence_ids)
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)  # (N, 8) integer codes [0-4]
        self.sentence_ids = sentence_ids
        self.sentences = sentences
        self.review_ids = review_ids

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "sentence_id": self.sentence_ids[idx],
            "sentence": self.sentences[idx],
            "review_id": self.review_ids[idx],
        }


# ==============================================================================
# 3. Data Loading & Preprocessing
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Ensure complete reproducibility across runs (Slide 25)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_dataset(
    data_dir: Path,
    labels_path: Path,
    allow_unreviewed: bool = False,
    skip_needs_review: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Load embeddings, align with labels via sentence_id, and filter exclusions.

    Returns:
        embeddings: (N, 256) float32 numpy array
        labels: (N, 8) int64 numpy array (states 0-4 for each aspect)
        sentence_ids: list of N sentence_id strings
        sentences: list of N sentence texts
        review_ids: list of N review_id strings (for group splitting)
    """
    npz_path = data_dir / "hotel_embeddings_256.npz"
    csv_path = data_dir / "hotel_sentences_2000.csv"

    if not npz_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {npz_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Sentences CSV not found: {csv_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels JSONL not found: {labels_path}")

    # 1. Load precomputed embeddings
    with np.load(npz_path, allow_pickle=False) as data:
        all_embeddings = data["embeddings"]  # (2000, 256)
        all_sentence_ids = data["sentence_ids"].tolist()
        id_to_embed_idx = {sid: i for i, sid in enumerate(all_sentence_ids)}

    # 2. Load sentence-to-review mapping from CSV
    sid_to_review_id: dict[str, str] = {}
    sid_to_sentence: dict[str, str] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["sentence_id"]
            sid_to_review_id[sid] = row.get("review_id", sid)
            sid_to_sentence[sid] = row.get("sentence", "")

    # 3. Load and validate labels
    valid_embeddings: list[np.ndarray] = []
    valid_labels: list[list[int]] = []
    valid_sids: list[str] = []
    valid_sentences: list[str] = []
    valid_review_ids: list[str] = []

    total_records = 0
    excluded_count = 0
    unreviewed_count = 0
    needs_review_count = 0

    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            rec = json.loads(line)
            sid = rec.get("sentence_id", "")

            # Filter: Check exclusion flag (Slide 16, 22)
            if rec.get("exclude") is True or rec.get("review_status") == "excluded":
                excluded_count += 1
                continue

            # Filter: Check review status (Slide 21: AI suggestions must be reviewed / audited)
            status = rec.get("review_status", "")
            is_audited_or_confirmed = status in ("ai_audited", "human_confirmed", "confirmed", "reviewed")
            if not is_audited_or_confirmed and not allow_unreviewed:
                unreviewed_count += 1
                continue

            # Optional: Check needs_review flag
            if rec.get("needs_review") is True:
                needs_review_count += 1
                if skip_needs_review:
                    continue

            # Check states field
            states = rec.get("states")
            if not isinstance(states, list) or len(states) != NUM_ASPECTS:
                continue
            if any(s not in STATES for s in states):
                continue

            # Lookup embedding
            if sid not in id_to_embed_idx:
                print(f"[Warning] sentence_id {sid} not found in embeddings file! Skipping.")
                continue

            embed_idx = id_to_embed_idx[sid]
            valid_embeddings.append(all_embeddings[embed_idx])
            valid_labels.append(states)
            valid_sids.append(sid)
            valid_sentences.append(rec.get("sentence") or sid_to_sentence.get(sid, ""))
            valid_review_ids.append(sid_to_review_id.get(sid, sid))

    print("=" * 70)
    print(f"Data Loading Summary ({labels_path.name}):")
    print(f"  Total records in file:        {total_records}")
    print(f"  Excluded records (dropped):   {excluded_count}")
    if unreviewed_count > 0:
        print(f"  Unreviewed AI suggestions:    {unreviewed_count} (skipped; use --allow-unreviewed to include)")
    print(f"  Records with needs_review:    {needs_review_count}")
    print(f"  Final valid training records: {len(valid_labels)}")
    print("=" * 70)

    if len(valid_labels) == 0:
        raise ValueError(
            "No valid labeled records loaded! If using unreviewed AI suggestions for quick testing, "
            "please add the flag: --allow-unreviewed"
        )

    return (
        np.array(valid_embeddings, dtype=np.float32),
        np.array(valid_labels, dtype=np.int64),
        valid_sids,
        valid_sentences,
        valid_review_ids,
    )


def split_by_review_id(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sentence_ids: list[str],
    sentences: list[str],
    review_ids: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[HotelAspectDataset, HotelAspectDataset, HotelAspectDataset]:
    """Split dataset into Train, Val, and Test sets grouped by review_id (Slide 22).

    This strictly guarantees that sentences from the same review are not split
    across train and validation/test sets, preventing data leakage.
    """
    rng = random.Random(seed)

    # Group sample indices by review_id
    review_to_indices: dict[str, list[int]] = {}
    for idx, rid in enumerate(review_ids):
        review_to_indices.setdefault(rid, []).append(idx)

    unique_reviews = list(review_to_indices.keys())
    rng.shuffle(unique_reviews)

    total_n = len(embeddings)
    target_test_n = max(1, int(total_n * test_ratio)) if test_ratio > 0 else 0
    target_val_n = max(1, int(total_n * val_ratio)) if val_ratio > 0 else 0

    test_indices: list[int] = []
    val_indices: list[int] = []
    train_indices: list[int] = []

    for rid in unique_reviews:
        idxs = review_to_indices[rid]
        if len(test_indices) < target_test_n:
            test_indices.extend(idxs)
        elif len(val_indices) < target_val_n:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    # Edge case: If training set ended up empty due to small sample size
    if len(train_indices) == 0:
        train_indices = val_indices
        val_indices = test_indices[: max(1, len(test_indices) // 2)]
        test_indices = test_indices[len(val_indices) :]

    def make_dataset(indices: list[int]) -> HotelAspectDataset:
        return HotelAspectDataset(
            embeddings=embeddings[indices],
            labels=labels[indices],
            sentence_ids=[sentence_ids[i] for i in indices],
            sentences=[sentences[i] for i in indices],
            review_ids=[review_ids[i] for i in indices],
        )

    train_ds = make_dataset(train_indices)
    val_ds = make_dataset(val_indices)
    test_ds = make_dataset(test_indices)

    print(f"Data Split Summary (Grouped by review_id, seed={seed}):")
    print(f"  Train set: {len(train_ds)} sentences ({len(set(train_ds.review_ids))} unique reviews)")
    print(f"  Val set:   {len(val_ds)} sentences ({len(set(val_ds.review_ids))} unique reviews)")
    print(f"  Test set:  {len(test_ds)} sentences ({len(set(test_ds.review_ids))} unique reviews)")
    print("-" * 70)

    return train_ds, val_ds, test_ds


# ==============================================================================
# 4. Model Architecture (Slides 24 & 25)
# ==============================================================================

class HotelReviewMLP(nn.Module):
    """Multilayer Perceptron for Multi-Aspect Hotel Sentiment Classification.

    Architecture (Slide 24):
        Input: 256-dim embedding
        Hidden Layers: configurable widths e.g. [128, 64] with ReLU and Dropout
        Output: 40 raw logits, reshaped into (batch_size, 8, 5)
        Logits shape: (batch_size, 8 aspects, 5 states)

    Architectures to compare (Slide 25):
        Run A: hidden_dims = [64]
        Run B: hidden_dims = [128, 64] (default)
        Run C: hidden_dims = [128, 64, 32]
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        num_aspects: int = NUM_ASPECTS,
        num_states: int = NUM_STATES,
        dropout: float = 0.2,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_aspects = num_aspects
        self.num_states = num_states
        self.output_dim = num_aspects * num_states  # 8 * 5 = 40

        layers: list[nn.Module] = []
        in_features = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = h_dim

        # Final projection to 40 logits
        layers.append(nn.Linear(in_features, self.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, 256) float tensor

        Returns:
            logits: (batch_size, 8, 5) float tensor
        """
        raw_scores = self.network(x)  # (batch_size, 40)
        logits = raw_scores.reshape(-1, self.num_aspects, self.num_states)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper returning predicted state codes and softmax probabilities.

        Returns:
            pred_states: (batch_size, 8) long tensor of states [0-4]
            probs: (batch_size, 8, 5) float tensor of softmax probabilities per aspect
        """
        logits = self.forward(x)  # (B, 8, 5)
        # Apply softmax per aspect (Slide 24: "one softmax per aspect, not over all 40")
        probs = torch.softmax(logits, dim=-1)
        pred_states = torch.argmax(probs, dim=-1)
        return pred_states, probs

    def count_parameters(self) -> int:
        """Return total number of trainable parameters (Slide 25 requirement)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# 5. Class Imbalance & Loss Function (Slide 27)
# ==============================================================================

def compute_class_weights(
    train_labels: np.ndarray,
    damping_factor: float = 0.5,
    class_0_weight_cap: float = 0.25,
) -> torch.Tensor:
    """Compute balanced class weights to address the severe dominance of state 0 ('absent').

    Slide 27 warning:
        "Predicting 'absent' often can produce high accuracy while missing useful opinions."
    By downweighting state 0 and boosting states 1-4, the MLP learns to actually detect
    hotel aspects instead of trivially outputting all-zeros.
    """
    flat_labels = train_labels.reshape(-1)
    counts = np.bincount(flat_labels, minlength=NUM_STATES).astype(np.float32)
    total_samples = len(flat_labels)

    # Inverse frequency with square-root smoothing (damping)
    weights = np.zeros(NUM_STATES, dtype=np.float32)
    for c in range(NUM_STATES):
        if counts[c] > 0:
            weights[c] = (total_samples / (NUM_STATES * counts[c])) ** damping_factor
        else:
            weights[c] = 1.0

    # Specifically dampen class 0 ('absent') to encourage aspect detection
    weights[0] = min(weights[0], class_0_weight_cap)

    # Normalize weights so that mean is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_per_aspect_class_weights(
    train_labels: np.ndarray,
    damping_factor: float = 0.5,
    class_0_weight_cap: float = 0.25,
    rare_boost: float = 1.15,
) -> torch.Tensor:
    """Per-aspect (8, 5) CrossEntropy weights.

    Aspects like food / sound are much sparser than service / location, so a single
    global 5-class weight vector under-trains rare heads. Still uses CrossEntropyLoss
    (Slide 26); only the class-weight schedule is finer-grained (Slide 27 imbalance).
    Also slightly boosts rare states 2 (neutral) and 4 (mixed).
    """
    n_aspects = train_labels.shape[1]
    out = np.ones((n_aspects, NUM_STATES), dtype=np.float32)
    for a in range(n_aspects):
        counts = np.bincount(train_labels[:, a], minlength=NUM_STATES).astype(np.float32)
        total = float(counts.sum())
        w = np.zeros(NUM_STATES, dtype=np.float32)
        for c in range(NUM_STATES):
            if counts[c] > 0:
                w[c] = (total / (NUM_STATES * counts[c])) ** damping_factor
            else:
                w[c] = 1.0
        w[0] = min(w[0], class_0_weight_cap)
        # Rare polarity classes in this corpus: neutral & mixed
        w[2] *= rare_boost
        w[4] *= rare_boost
        w = w / max(w.mean(), 1e-8)
        out[a] = w
    return torch.tensor(out, dtype=torch.float32)


# ==============================================================================
# 6. Evaluation Metrics (Slide 27)
# ==============================================================================

@dataclass
class EvaluationMetrics:
    loss: float
    overall_accuracy: float
    mention_precision: float
    mention_recall: float
    mention_f1: float
    macro_f1: float
    micro_f1: float
    non_absent_macro_f1: float
    per_aspect_acc: dict[str, float]
    per_aspect_mention_f1: dict[str, float]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: float = 0.0,
) -> EvaluationMetrics:
    """Compute detailed evaluation metrics matching Slide 27:

    1. Overall accuracy: fraction of all (N × 8) correct state predictions.
    2. Mention Detection (Binary: state != 0 vs state == 0):
       Precision, Recall, F1 for identifying whether an aspect was discussed.
    3. End-to-end sentiment + aspect:
       Micro-F1 and Macro-F1 across all 5 classes.
    4. Per-aspect results:
       Individual accuracy and mention-F1 for each of the 8 aspects.
    """
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    overall_acc = float((y_true_flat == y_pred_flat).mean())

    # 1. Mention Detection (Binary: 0=absent, 1=mentioned)
    mention_true = (y_true_flat != 0).astype(int)
    mention_pred = (y_pred_flat != 0).astype(int)

    tp = int(((mention_true == 1) & (mention_pred == 1)).sum())
    fp = int(((mention_true == 0) & (mention_pred == 1)).sum())
    fn = int(((mention_true == 1) & (mention_pred == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mention_f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # 2. Multi-class F1 Scores
    if HAS_SKLEARN:
        macro_f1 = float(f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0))

        # Macro-F1 over non-absent classes [1, 2, 3, 4]
        non_absent_mask = np.isin(y_true_flat, [1, 2, 3, 4]) | np.isin(y_pred_flat, [1, 2, 3, 4])
        if non_absent_mask.any():
            non_absent_f1 = float(
                f1_score(
                    y_true_flat[non_absent_mask],
                    y_pred_flat[non_absent_mask],
                    labels=[1, 2, 3, 4],
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            non_absent_f1 = 0.0
    else:
        macro_f1 = overall_acc
        micro_f1 = overall_acc
        non_absent_f1 = mention_f1

    # 3. Per-Aspect Metrics
    per_aspect_acc: dict[str, float] = {}
    per_aspect_mention_f1: dict[str, float] = {}

    for a_idx, aspect in enumerate(ASPECTS):
        a_true = y_true[:, a_idx]
        a_pred = y_pred[:, a_idx]
        per_aspect_acc[aspect] = float((a_true == a_pred).mean())

        # Aspect mention F1
        m_true = (a_true != 0).astype(int)
        m_pred = (a_pred != 0).astype(int)
        tp_a = int(((m_true == 1) & (m_pred == 1)).sum())
        fp_a = int(((m_true == 0) & (m_pred == 1)).sum())
        fn_a = int(((m_true == 1) & (m_pred == 0)).sum())
        p_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        r_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        f1_a = (2 * p_a * r_a) / (p_a + r_a) if (p_a + r_a) > 0 else 0.0
        per_aspect_mention_f1[aspect] = float(f1_a)

    return EvaluationMetrics(
        loss=loss,
        overall_accuracy=overall_acc,
        mention_precision=prec,
        mention_recall=rec,
        mention_f1=mention_f1,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        non_absent_macro_f1=non_absent_f1,
        per_aspect_acc=per_aspect_acc,
        per_aspect_mention_f1=per_aspect_mention_f1,
    )


# ==============================================================================
# 7. Trainer & Evaluation Routine
# ==============================================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[EvaluationMetrics, np.ndarray, np.ndarray]:
    """Evaluate model over a dataloader.

    Returns metrics, all y_true, and all y_pred.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch["x"].to(device)
            y = batch["y"].to(device)

            logits = model(X).reshape(-1, NUM_ASPECTS, NUM_STATES)
            if isinstance(criterion, nn.Module):
                loss = criterion(logits.reshape(-1, NUM_STATES), y.reshape(-1))
            else:
                loss = criterion(logits, y)

            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(preds)

    avg_loss = total_loss / max(1, total_batches)
    y_true_mat = np.vstack(all_y_true)
    y_pred_mat = np.vstack(all_y_pred)
    metrics = compute_metrics(y_true_mat, y_pred_mat, loss=avg_loss)
    return metrics, y_true_mat, y_pred_mat


def train_model(
    model: nn.Module,
    train_ds: HotelAspectDataset,
    val_ds: HotelAspectDataset,
    args: argparse.Namespace,
    device: torch.device,
    save_dir: Path,
) -> dict[str, Any]:
    """Execute training loop with validation checkpointing and metric logging."""
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # BatchNorm requires batch size > 1 during training.
        drop_last=bool(getattr(args, "batch_norm", False)) or len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 1. Setup Loss Criterion (Slide 26: CrossEntropyLoss)
    # label_smoothing is still CrossEntropyLoss (regularized CE), fully slide-compliant.
    label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    per_aspect = bool(getattr(args, "per_aspect_weighting", False))

    if per_aspect:
        pa_weights = compute_per_aspect_class_weights(
            train_ds.y.numpy(),
            damping_factor=args.damping_factor,
            class_0_weight_cap=args.class_0_weight,
            rare_boost=float(getattr(args, "rare_class_boost", 1.15) or 1.15),
        ).to(device)  # (8, 5)
        print(f"Using PER-ASPECT class weights for CrossEntropy:\n{pa_weights.cpu().numpy().round(3)}")

        def multi_aspect_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # logits: (B, 8, 5); y: (B, 8)
            losses = []
            for a in range(NUM_ASPECTS):
                losses.append(
                    nn.functional.cross_entropy(
                        logits[:, a, :],
                        y[:, a],
                        weight=pa_weights[a],
                        label_smoothing=label_smoothing,
                    )
                )
            return torch.stack(losses).mean()

        criterion = multi_aspect_ce
    elif args.loss_weighting == "balanced":
        weights = compute_class_weights(
            train_ds.y.numpy(),
            damping_factor=args.damping_factor,
            class_0_weight_cap=args.class_0_weight,
        ).to(device)
        print(f"Using class weights for CrossEntropy: {weights.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 2. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mention_f1": [],
        "val_macro_f1": [],
        "val_acc": [],
    }

    best_val_score = -1.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting Training...")
    print(f"  Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Device: {device}")
    print(f"  Model Parameters: {model.count_parameters():,}")
    print("-" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Mention F1':^11} | {'Macro F1':^10} | {'Status':^8}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            X = batch["x"].to(device)
            y = batch["y"].to(device)  # (batch_size, 8)

            # Slide 26 exact snippet:
            logits = model(X).reshape(-1, NUM_ASPECTS, NUM_STATES)
            if per_aspect:
                loss = criterion(logits, y)
            else:
                loss = criterion(logits.reshape(-1, NUM_STATES), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        # Validation evaluation
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics.macro_f1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics.loss)
        history["val_mention_f1"].append(val_metrics.mention_f1)
        history["val_macro_f1"].append(val_metrics.macro_f1)
        history["val_acc"].append(val_metrics.overall_accuracy)

        # Use Macro-F1 + Mention-F1 composite score for checkpoint selection
        val_score = 0.6 * val_metrics.macro_f1 + 0.4 * val_metrics.mention_f1
        is_best = val_score > best_val_score

        status_str = ""
        if is_best:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0
            status_str = "⭐ Best"

            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": INPUT_DIM,
                    "hidden_dims": args.hidden_dims,
                    "num_aspects": NUM_ASPECTS,
                    "num_states": NUM_STATES,
                    "dropout": args.dropout,
                    "aspect_names": ASPECTS,
                    "state_names": STATES,
                },
                "training_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "best_epoch": best_epoch,
                "val_metrics": asdict(val_metrics),
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

        if epoch % 5 == 0 or is_best or epoch == 1:
            print(
                f"{epoch:^7d} | {avg_train_loss:^10.4f} | {val_metrics.loss:^10.4f} | "
                f"{val_metrics.overall_accuracy*100:^8.1f}% | {val_metrics.mention_f1:^11.3f} | "
                f"{val_metrics.macro_f1:^10.3f} | {status_str:^8}"
            )

    print("-" * 80)
    print(f"Training finished. Best epoch: {best_epoch} with val score: {best_val_score:.4f}")
    return history


# ==============================================================================
# 8. Error Analysis & Test Reporting (Slide 27)
# ==============================================================================

def perform_error_analysis(
    test_ds: HotelAspectDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Identify and print failure cases with text and expected labels (Slide 27)."""
    error_cases: list[dict[str, Any]] = []

    for i in range(len(test_ds)):
        true_states = y_true[i]
        pred_states = y_pred[i]
        mismatches = []

        for a_idx, aspect in enumerate(ASPECTS):
            if true_states[a_idx] != pred_states[a_idx]:
                mismatches.append({
                    "aspect": aspect,
                    "expected": f"{true_states[a_idx]} ({STATES[true_states[a_idx]]})",
                    "predicted": f"{pred_states[a_idx]} ({STATES[pred_states[a_idx]]})",
                })

        if mismatches:
            error_cases.append({
                "sentence_id": test_ds.sentence_ids[i],
                "sentence": test_ds.sentences[i],
                "num_errors": len(mismatches),
                "mismatches": mismatches,
            })

    # Sort by number of mismatched aspects descending
    error_cases.sort(key=lambda x: x["num_errors"], reverse=True)

    print("\n" + "=" * 80)
    print(f"Top {min(top_k, len(error_cases))} Error Analysis Cases (Slide 27 Requirement):")
    print("=" * 80)

    for rank, err in enumerate(error_cases[:top_k], 1):
        print(f"[{rank}] Sentence ID: {err['sentence_id']}")
        print(f"    Text: {err['sentence']}")
        print("    Discrepancies:")
        for m in err["mismatches"]:
            print(f"      - {m['aspect']:<22} Expected: {m['expected']:<14} | Predicted: {m['predicted']}")
        print()

    return error_cases


def plot_training_curves(history: dict[str, list[float]], save_path: Path) -> None:
    """Plot and save training/validation loss and metric curves (Slide 27)."""
    if not HAS_MATPLOTLIB:
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Metric curve
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro F1", color="#2ca02c")
    ax2.plot(epochs, history["val_mention_f1"], label="Val Mention F1", color="#d62728")
    ax2.set_title("Validation Metric Progression")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved training history curves to: {save_path.name}")


# ==============================================================================
# 9. Main Orchestration
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Hotel Review MLP Training Script")

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing embeddings and CSVs",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="labels_reviewed.jsonl",
        help="Path to JSONL labels (defaults to labels_reviewed.jsonl, falls back to labels_ai_suggestions.jsonl)",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow training on unreviewed AI suggestions (useful before human audit is complete)",
    )
    parser.add_argument(
        "--skip-needs-review",
        action="store_true",
        help="Skip rows flagged with needs_review=True",
    )

    # Architecture arguments (Slide 25)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer dimensions (e.g. 64 for Run A, 128 64 for Run B, 128 64 32 for Run C)",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--batch-norm", action="store_true", help="Enable BatchNorm1d")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--loss-weighting", choices=["balanced", "none"], default="balanced", help="Address class-0 dominance using inverse-frequency class weights (Slide 27)")
    parser.add_argument("--class-0-weight", type=float, default=0.25, help="Weight cap for state 0 (absent)")
    parser.add_argument("--damping-factor", type=float, default=0.5, help="Exponent for frequency weight damping")
    parser.add_argument("--per-aspect-weighting", action="store_true", help="Use per-aspect (8,5) CE weights for sparse heads (food/sound)")
    parser.add_argument("--rare-class-boost", type=float, default=1.15, help="Extra boost for rare states 2/4 under per-aspect weighting")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="CrossEntropy label smoothing (0-0.2 recommended)")

    # Splitting & Experiment tracking
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for GroupKFold review_id split only. Defaults to --seed. "
             "Set a fixed value (e.g. 42) when comparing training seeds on the same test set.",
    )
    parser.add_argument("--exp-name", type=str, default="", help="Experiment subfolder name")

    args = parser.parse_args()
    set_seed(args.seed)

    # 1. Resolve labels path (auto detection: audited -> reviewed -> suggestions)
    labels_path = args.data_dir / args.labels_file
    if not labels_path.exists():
        candidates = [
            args.data_dir / "labels_audited.jsonl",
            args.data_dir / "labels_reviewed.jsonl",
            args.data_dir / "labels_ai_suggestions.jsonl",
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                print(f"[Info] {labels_path.name} not found. Auto-detected and using {cand.name}.")
                labels_path = cand
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Could not find any labels file in {args.data_dir}")

    # 2. Setup output experiment directory
    exp_suffix = args.exp_name or f"mlp_{'_'.join(map(str, args.hidden_dims))}_seed{args.seed}"
    save_dir = args.data_dir / "experiments" / exp_suffix
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load dataset
    embeddings, labels, sids, sentences, review_ids = load_raw_dataset(
        data_dir=args.data_dir,
        labels_path=labels_path,
        allow_unreviewed=args.allow_unreviewed,
        skip_needs_review=args.skip_needs_review,
    )

    # 4. Group split by review_id (Slide 22)
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    print(f"Split seed (review_id groups): {split_seed} | Train init seed: {args.seed}")
    train_ds, val_ds, test_ds = split_by_review_id(
        embeddings=embeddings,
        labels=labels,
        sentence_ids=sids,
        sentences=sentences,
        review_ids=review_ids,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=split_seed,
    )

    # 5. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = HotelReviewMLP(
        input_dim=INPUT_DIM,
        hidden_dims=args.hidden_dims,
        num_aspects=NUM_ASPECTS,
        num_states=NUM_STATES,
        dropout=args.dropout,
        use_batch_norm=args.batch_norm,
    ).to(device)

    # 6. Train Model
    history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        args=args,
        device=device,
        save_dir=save_dir,
    )

    # 7. Plot Curves
    plot_training_curves(history, save_dir / "loss_curves.png")

    # 8. Final Held-Out Test Evaluation (Slide 27)
    best_checkpoint_path = save_dir / "best_model.pt"
    if best_checkpoint_path.exists():
        chk = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(chk["model_state_dict"])
        print(f"\nLoaded best model from epoch {chk['best_epoch']} for final test evaluation.")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    criterion_eval = nn.CrossEntropyLoss()
    test_metrics, y_test_true, y_test_pred = evaluate(model, test_loader, criterion_eval, device)

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION REPORT (Slide 27)")
    print("=" * 80)
    print(f"  Overall Accuracy:          {test_metrics.overall_accuracy*100:.2f}%")
    print(f"  Mention Detection Precision: {test_metrics.mention_precision:.3f}")
    print(f"  Mention Detection Recall:    {test_metrics.mention_recall:.3f}")
    print(f"  Mention Detection F1:        {test_metrics.mention_f1:.3f}")
    print(f"  End-to-End Macro F1:         {test_metrics.macro_f1:.3f}")
    print(f"  End-to-End Micro F1:         {test_metrics.micro_f1:.3f}")
    print(f"  Non-Absent Macro F1:         {test_metrics.non_absent_macro_f1:.3f}")
    print("-" * 80)
    print("Per-Aspect Performance:")
    print(f"  {'Aspect':<25} | {'Accuracy':^10} | {'Mention F1':^12}")
    print("  " + "-" * 52)
    for aspect in ASPECTS:
        acc = test_metrics.per_aspect_acc.get(aspect, 0.0) * 100
        mf1 = test_metrics.per_aspect_mention_f1.get(aspect, 0.0)
        print(f"  {aspect:<25} | {acc:^9.1f}% | {mf1:^12.3f}")
    print("=" * 80)

    # 9. Top Error Cases (Slide 27)
    error_cases = perform_error_analysis(test_ds, y_test_true, y_test_pred, top_k=5)

    # 10. Save Metrics Artifact
    metrics_summary = {
        "experiment": exp_suffix,
        "parameters": model.count_parameters(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "test_metrics": asdict(test_metrics),
        "top_error_cases": error_cases,
    }
    with (save_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved test metrics and error report to: {save_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
