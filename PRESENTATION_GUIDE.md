# Project 1 Presentation Guide: The 3-Tier MNIST Machine Learning Trilogy
> **Deep Learning Course 2609 — Project 1**  
## Classical ML (Logistic Regression) vs. Deep Feed-Forward (MLP) vs. Spatial Deep Learning (CNN)

This document provides a comprehensive, presentation-ready comparison of **all three models** implemented in this course project.

---

## 1. The 3-Tier Architecture Progression

```
Tier 1: Linear Machine Learning      Tier 2: Multi-Layer Perceptron (MLP)     Tier 3: Convolutional Neural Network (CNN)
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐   ┌────────────────────────────────────────┐
│ Scikit-Learn Logistic Regression│   │ PyTorch Deep Feed-Forward Network   │   │ PyTorch Convolutional Neural Network   │
│                                 │   │                                     │   │                                        │
│ • No hidden layers              │   │ • 2 Hidden Dense Layers (256 -> 64) │   │ • 2 Conv2D + MaxPool2D feature stages  │
│ • 1D flattened input (784)      │   │ • 1D flattened input (784)          │   │ • 2D spatial feature preservation      │
│ • 7,850 parameters              │   │ • 218,058 parameters                │   │ • 97,706 parameters (weight sharing!)  │
│ • Test Accuracy: 92.59%         │   │ • Test Accuracy: 97.68%             │   │ • Test Accuracy: 99.11%                │
└─────────────────────────────────┘   └─────────────────────────────────────┘   └────────────────────────────────────────┘
```

---

## 2. Directory Layout & Symmetric File Pairs

All files in [`mnist-course-project/`](.) are organized into clean, parallel sets:

```
mnist-course-project/
│
├── 🧠 [TIER 3: PYTORCH CNN]
│   ├── train_cnn.py                  # PyTorch CNN training pipeline (Conv2D -> MaxPool -> Dense)
│   ├── evaluate_cnn.py               # Generates CNN confusion matrix & metrics
│   ├── weights.js                    # Exported CNN weights (~830 KB, ~97k parameters)
│   ├── models/mnist_cnn.pt           # PyTorch CNN checkpoint
│   └── plots/confusion_matrix.png    # 10x10 annotated CNN confusion matrix (99.11% Acc)
│
├── ⚡ [TIER 2: PYTORCH MLP]
│   ├── train_mlp.py                  # PyTorch MLP training pipeline (Linear 784->256->64->10)
│   ├── evaluate_mlp.py               # Generates MLP confusion matrix & curves
│   ├── mlp_weights.js                # Exported MLP weights (~880 KB, ~218k parameters)
│   ├── models/mnist_mlp.pt           # PyTorch MLP checkpoint
│   └── plots/mlp_confusion_matrix.png# 10x10 annotated MLP confusion matrix (97.68% Acc)
│
├── 📊 [TIER 1: SCIKIT-LEARN LOGISTIC REGRESSION]
│   ├── train_sklearn.py              # Scikit-Learn training script (Multinomial Logistic Regression)
│   ├── evaluate_sklearn.py           # Evaluates Scikit-Learn model on 10k test images
│   ├── sklearn_weights.js            # Exported linear weights (~42 KB, ~7.8k parameters)
│   ├── models/sklearn_logistic_regression.joblib  # Serialized Scikit-Learn checkpoint
│   └── plots/sklearn_confusion_matrix.png # Scikit-Learn confusion matrix (92.59% Acc)
│
└── 🌐 [WEB APPLICATION & LIVE DEMO (PORT 8791)]
    ├── index.html                    # Web UI supporting all 3 models + 3-way live comparison
    ├── style.css                     # Responsive styling & distinct model theme colors
    └── app.js                        # Client-side inference engines for all 3 models
```

---

## 3. Comprehensive Model Comparison Matrix (Slide Cheat-Sheet)

| Feature | 📊 Model 1: Scikit-Learn | ⚡ Model 2: PyTorch MLP | 🧠 Model 3: PyTorch CNN |
| :--- | :--- | :--- | :--- |
| **Model Type** | Generalized Linear Model | Deep Feed-Forward (Dense) | Deep Convolutional Network |
| **Layers** | 0 hidden (1 Linear layer) | 2 hidden (`Dense 256` $\rightarrow$ `64`) | 2 Conv stages + 2 Dense layers |
| **Input Shape** | 1D vector (`784`) | 1D vector (`784`) | 2D matrix (`1, 28, 28`) |
| **Total Parameters** | **~7,850** | ~218,058 | **~97,706** *(55% fewer than MLP!)* |
| **Payload Size** | **~42 KB** | ~880 KB | ~830 KB |
| **Test Accuracy** | **92.59%** | **97.68%** | **99.11%** |
| **Macro F1-Score** | 0.9248 | 0.9767 | **0.9911** |
| **Inference Time** | **~0.1 ms** | ~0.4 ms | ~7 ms |
| **Spatial Invariance** | None | Low (overfits pixel locations) | **High (translation & scale robust)** |
| **Key Advantage** | Ultra-lightweight & convex | Non-linear feature combination | Spatial pattern extraction & weight sharing |

---

## 4. The Killer Presentation Talking Point: Why CNN Beats MLP With FEWER Parameters!

A classic question professors love to ask:
> *"Why does the MLP have 218,000 parameters while the CNN has only 97,000 parameters, yet the CNN achieves much higher accuracy?"*

### The Answer for Your Slides:
1. **Full Connectivity vs. Local Connectivity**:
   - The first layer of the MLP connects all 784 pixels to 256 neurons ($784 \times 256 = 200,704$ weights alone!). It assumes any pixel can interact equally with any other pixel, even if they are on opposite sides of the canvas.
2. **Weight Sharing in Convolutions**:
   - The CNN's first layer uses $3 \times 3$ filters. Each filter only has $3 \times 3 = 9$ weights! That same 9-weight filter slides across the entire $28 \times 28$ image to detect edges everywhere.
3. **Inductive Bias for Images**:
   - Images have local correlation (neighboring pixels form lines, curves, and corners). CNN exploits this spatial structure naturally, while MLP has to blindly learn it from scratch.

---

## 5. Live Interactive Demo Checklist (`http://localhost:8791`)

1. Open [http://localhost:8791](http://localhost:8791) on your screen.
2. Show the **Model Selector Bar**:
   - **🧠 PyTorch CNN (99.11%)**: Deep learning mode.
   - **⚡ PyTorch MLP (97.68%)**: Dense network mode.
   - **📊 Scikit-Learn ML (92.59%)**: Linear model mode.
3. Click **"⚖️ Compare All 3"**:
   - Draw a clean digit (e.g. `7`): All three models will agree with high confidence.
   - Draw a slightly unusual, cursive, or off-center digit (e.g. tilted `4` or loop `2`):
     - **CNN** will confidently recognize it (thanks to 2D convolutions + Center-of-Mass).
     - **MLP** will have reduced confidence.
     - **Logistic Regression** might misclassify it, clearly demonstrating why Deep Learning replaced linear models for computer vision.
