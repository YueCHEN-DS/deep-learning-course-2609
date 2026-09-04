# Project 1: Handwritten Digit Recognition (MNIST)
> **Deep Learning Course 2609 — Project 1**  
> **Repository**: [https://github.com/YueCHEN-DS/deep-learning-course-2609](https://github.com/YueCHEN-DS/deep-learning-course-2609)  
> **Live Demo**: [http://localhost:8791](http://localhost:8791)

A complete machine learning course project comparing three distinct model architectures on the MNIST dataset:
1. **Classical Linear ML**: Scikit-Learn Multinomial Logistic Regression (**92.59%** Acc)
2. **Deep Feed-Forward Network**: PyTorch Multi-Layer Perceptron (MLP) (**97.68%** Acc)
3. **Spatial Deep Learning**: PyTorch 2-Stage Convolutional Neural Network (CNN) (**99.11%** Acc)

Includes client-side browser deployment with live multi-model inference and side-by-side comparison on port 8791.

---

## 1. Directory Structure

All frontend application files, PyTorch training scripts, Scikit-Learn scripts, models, and plots are self-contained in this single folder:

```
mnist-course-project/
├── index.html                # Multi-model Web Application UI (Port 8791)
├── style.css                 # Web Application styling & themes
├── app.js                    # Dual-inference engine (CNN + Linear ML)
├── weights.js                # PyTorch CNN weights (99.11% accuracy)
├── mlp_weights.js            # PyTorch MLP weights (97.68% accuracy)
├── sklearn_weights.js        # Scikit-Learn weights (92.59% accuracy)
├── train_cnn.py              # PyTorch CNN model training script
├── train_mlp.py              # PyTorch MLP model training script
├── train_sklearn.py          # Scikit-Learn classical ML training script
├── evaluate_cnn.py           # Evaluation script for CNN
├── evaluate_mlp.py           # Evaluation script for MLP
├── evaluate_sklearn.py       # Evaluation script for Scikit-Learn
├── export_to_web.py          # Weight exporter
├── models/                   # Model checkpoints (.pt, .joblib) & text reports
├── plots/                    # Confusion matrices and training curves
├── data/                     # MNIST raw dataset
├── README.md                 # Project documentation
└── PRESENTATION_GUIDE.md     # Course presentation / Pre cheat-sheet & talking points
```

### Web Application:
Running live at: **[http://localhost:8791](http://localhost:8791)**

```bash
cd mnist-course-project
python3 -m http.server 8791
```

---

## 2. Model Architecture & Parameters

```
Input: (batch_size, 1, 28, 28)
  │
  ├── Conv2D (1 -> 16, kernel=3, padding=1) + ReLU
  ├── MaxPool2D (2, 2)                         ==> (16, 14, 14)
  │
  ├── Conv2D (16 -> 32, kernel=3, padding=1) + ReLU
  ├── MaxPool2D (2, 2)                         ==> (32, 7, 7) = 1,568 features
  │
  ├── Flatten
  ├── Linear (1568 -> 96) + ReLU + Dropout(0.2)
  └── Linear (96 -> 10)                        ==> 10 Class Logits
```
- **Total Parameters**: ~97,700
- **Optimizer**: Adam ($\text{lr} = 0.002$)
- **Loss Function**: Cross-Entropy Loss
- **Data Augmentation**: Random rotation ($\pm 12^\circ$), translation ($\pm 8\%$), and scale ($0.9 - 1.1\times$).

---

## 3. Results & Evaluation Metrics

- **Final Test Accuracy**: **99.11%** (10,000 test images)
- **Training Time**: ~34 seconds on Apple Silicon GPU (`mps`).

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.9949 | 0.9969 | 0.9959 | 980 |
| **1** | 0.9956 | 0.9938 | 0.9947 | 1,135 |
| **2** | 0.9828 | 0.9952 | 0.9889 | 1,032 |
| **3** | 0.9921 | 0.9950 | 0.9936 | 1,010 |
| **4** | 0.9949 | 0.9868 | 0.9908 | 982 |
| **5** | 0.9922 | 0.9933 | 0.9927 | 892 |
| **6** | 0.9948 | 0.9906 | 0.9927 | 958 |
| **7** | 0.9912 | 0.9844 | 0.9878 | 1,028 |
| **8** | 0.9918 | 0.9908 | 0.9913 | 974 |
| **9** | 0.9812 | 0.9841 | 0.9827 | 1,009 |
| **Overall** | **0.9911** | **0.9911** | **0.9911** | **10,000** |

---

## 4. How to Run

Activate your conda environment first:
```bash
conda activate pytorch_env
```

### Run Training:
```bash
python train.py
```

### Generate Plots & Metrics:
```bash
python evaluate.py
```
Generated figures will be saved in `./plots/`.

### Export Weights to the Web App (Optional):
```bash
python export_to_web.py ../digit-recognizer/weights.js
```
