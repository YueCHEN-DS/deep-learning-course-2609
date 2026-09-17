# Project 2: Hotel Review Aspect-Based Sentiment Analysis (MLP)
> **Deep Learning Course 2609 — Project 2**  
> **Repository**: [https://github.com/YueCHEN-DS/deep-learning-course-2609](https://github.com/YueCHEN-DS/deep-learning-course-2609)  
> **Live Demo**: [http://localhost:8792](http://localhost:8792)

A fine-grained, multi-task deep learning system classifying Chinese hotel reviews across **8 aspects × 5 sentiment states** (40 output logits). Incorporates a 4-generation benchmark evolution progression, FastAPI server, and a modern glassmorphism dark-theme Web Studio with interactive ground-truth editing and 1-click retraining.

---

## 1. Directory Structure

All application files, PyTorch training pipelines, dataset files, models, and benchmark reports are organized cleanly:

```
course_project2_hotel_review_sentiment_analysis/
├── index.html                             # Web Application UI & Studio (Port 8792)
├── style.css                              # Glassmorphism dark-theme styling
├── app.js                                 # Web interactivity, API client, & annotation studio
├── server.py                              # FastAPI backend (Inference, Probe, Dataset Studio, Retrain)
├── train_mlp.py                           # PyTorch multi-task MLP training & evaluation pipeline
├── export_to_web.py                       # Weight exporter
├── load_dataset.py                        # Dataset loading utility
├── get_qwen_embedding.py                  # Qwen 256-dim embedding demo
├── annotate_with_deepseek.py              # DeepSeek LLM assisted annotation tool
│
├── hotel_sentences_2000.csv               # 2,000 candidate sentences & context metadata
├── hotel_embeddings_256.npz               # 2,000 sentence 256-dim Qwen embeddings
├── hotel_demo_reviews_30.csv              # 30 full review demo examples
├── labels_final_2000_v4_mlp.jsonl         # Audited ground-truth dataset (2,000 items)
├── labels_final_2000_v4_mlp_notes.md      # Annotation guidelines & audit rules
├── records_to_review.csv                  # Flagged human review records
├── requirements.txt                       # Project dependencies
├── .gitignore                             # Git ignore rules (protects API keys)
├── api_keys.example.txt                   # API Key configuration template
├── README.md                              # Project documentation
├── PRESENTATION_GUIDE.md                  # Defense & presentation cheat-sheet
│
└── experiments/                           # 4-Generation Benchmark checkpoints & reports
    ├── run_B_default_128_64/              # Gen 1 Baseline (100 samples)
    ├── run_model2_optimized_500data/      # Gen 2 Optimized (500 samples)
    ├── run_model3_champion_1000v3/        # Gen 3 Champion (1,000 samples)
    ├── run_model4_champion_2000v4/        # Gen 4 Full Champion (2,000 samples)
    ├── run_model4_interactive_retrained/  # Live interactive retrained checkpoint
    ├── baseline_100data_report.md         # Gen 1 Evaluation report
    ├── model2_500data_report.md           # Gen 2 Evaluation report
    ├── model3_1000data_report.md          # Gen 3 Evaluation report
    ├── model4_2000data_report.md          # Gen 4 Evaluation report
    └── mlp_optimization_analysis.md       # 4-Generation Ablation & Evolution report
```

---

## 2. Quick Start

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Configure API Keys (Optional)
To generate embeddings for custom review sentences or use AI annotation:
```bash
cp api_keys.example.txt api_keys.txt
# Edit api_keys.txt with your DashScope/DeepSeek keys
```
Or set environment variables:
```bash
export DASHSCOPE_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

### Step 3: Launch Web Studio
```bash
python server.py --host 127.0.0.1 --port 8792
```
Open **[http://localhost:8792](http://localhost:8792)** in your browser.
- **Default login**: `teacher` / `admin123`
- **Main Tabs**:
  1. **📊 Hotel Review Sentiment**: Multi-sentence review decomposition with $[-100, +100]$ score aggregation.
  2. **🔍 Single Sentence Probe**: Real-time aspect probability breakdown.
  3. **📈 Model Metrics & Benchmark**: 4-generation comparison table, radar charts, loss curves.
  4. **📁 Training Data Studio**: 2,000-sample explorer, disagreement filter, in-browser ground truth editor, and 1-click retraining.

---

## 3. Model Training & Evaluation

To reproduce the Gen 4 Champion model:
```bash
python train_mlp.py --data-version v4 --epochs 60 --lr 0.0008 --hidden-dims 384 192
```

To run evaluation only on the active model:
```bash
python train_mlp.py --eval-only
```

---

## 4. 4-Generation Evolution Summary

| Metric | Gen 1 (100 Data) | Gen 2 (500 Data) | Gen 3 (1,000 Data) | Gen 4 Champion (2,000 Data) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 50.00% | 76.35% | 85.31% | **87.64%** |
| **Mention F1** | 0.416 | 0.626 | 0.714 | **0.790** |
| **Macro F1** | 0.284 | 0.394 | 0.497 | **0.554** |
| **Non-Absent Macro F1** | 0.120 | 0.280 | 0.391 | **0.458** |
| **Mention Recall** | 61.5% | 73.2% | 76.7% | **82.5%** |
| **Mention Precision** | 31.4% | 54.7% | 66.8% | **75.8%** |
| **Test Loss** | 1.351 | 0.684 | 0.492 | **0.399** |

---

## 5. Aspect Taxonomy & Polarity Aggregation

### 8 Aspects
`cleanliness`, `service`, `location`, `facilities`, `room_comfort`, `sound_insulation_noise`, `food`, `value`.

### 5 States
`0 absent`, `1 negative`, `2 neutral`, `3 positive`, `4 mixed`.

### Review Aggregation Formula
For aspect $a$ across $n_a$ mentioned sentences ($s_{ia} \in \{+1, -1, 0\}$):

$$S_a = 100 \times \frac{\sum_{i=1}^{n_a} s_{ia}}{n_a}$$
