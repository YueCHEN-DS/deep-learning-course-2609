# Deep Learning Course 2609

Repository for coursework, deep learning models, and interactive web applications for **Deep Learning Course 2609**.

---

## Course Projects

### [📁 Project 1: Handwritten Digit Recognition (`course_project1_handwritting_recognition`)](./course_project1_handwritting_recognition/)
A complete machine learning project comparing three distinct model architectures on the MNIST dataset:
- **Classical Linear ML**: Scikit-Learn Multinomial Logistic Regression (**92.59%** accuracy)
- **Deep Feed-Forward Network**: PyTorch Multi-Layer Perceptron (MLP) (**97.68%** accuracy)
- **Spatial Deep Learning**: PyTorch 2-Stage Convolutional Neural Network (CNN) (**99.11%** accuracy)
- **Web Application**: Client-side interactive canvas with real-time multi-model inference and 3-way live comparison on port **8791**.

Documentation:
- [Project 1 README](./course_project1_handwritting_recognition/README.md)
- [Presentation Guide & Defense Prep](./course_project1_handwritting_recognition/PRESENTATION_GUIDE.md)

---

### [📁 Project 2: Hotel Review Aspect-Based Sentiment Analysis (`course_project2_hotel_review_sentiment_analysis`)](./course_project2_hotel_review_sentiment_analysis/)
A fine-grained, multi-task NLP sentiment analysis system classifying Chinese hotel reviews across **8 aspects × 5 sentiment states** (40-output logits):
- **Core Architecture**: Deep PyTorch MLP (`256 → 384 → 192 → 40`) trained on 256-dimensional Qwen text embeddings.
- **4-Generation Evolution Benchmark**:
  - **Gen 1** (100 baseline records): `50.00%` Acc / `0.416` Mention-F1 / `0.284` Macro-F1
  - **Gen 2** (500 records): `76.35%` Acc / `0.626` Mention-F1 / `0.394` Macro-F1
  - **Gen 3** (1,000 records): `85.31%` Acc / `0.714` Mention-F1 / `0.497` Macro-F1
  - **Gen 4 Champion** (2,000 full audited records): **`87.64%` Acc / `0.790` Mention-F1 / `0.554` Macro-F1** (with 0.399 Test Loss)
- **FastAPI & Glassmorphism Web Studio** (Running on port **8792**):
  - **Review Sentiment Analyzer**: Multi-sentence review decomposition with $[-100, +100]$ score aggregation: $S_a = 100 \times \frac{\sum s_{ia}}{n_a}$.
  - **Single Sentence Inspector**: Live aspect detection probe with probability breakdown.
  - **Benchmark Evolution Dashboard**: Detailed Gen 1 ~ Gen 4 metric progression tables, per-aspect F1 charts, and training loss curves.
  - **Training Dataset Studio**: 2,000-sample visualizer, model disagreement filter, in-browser ground truth editor, and 1-click hot retraining workflow.

Documentation:
- [Project 2 README](./course_project2_hotel_review_sentiment_analysis/README.md)
- [Presentation Guide & Defense Prep](./course_project2_hotel_review_sentiment_analysis/PRESENTATION_GUIDE.md)
- [Annotation Notes & Multi-Agent Audit](./course_project2_hotel_review_sentiment_analysis/labels_final_2000_v4_mlp_notes.md)
- [MLP Optimization & Ablation Report](./course_project2_hotel_review_sentiment_analysis/experiments/mlp_optimization_analysis.md)
