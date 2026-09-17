# Project 2 Presentation Guide: The 4-Generation Hotel Review Sentiment MLP Evolution
> **Deep Learning Course 2609 — Project 2**  
> ## Fine-Grained Aspect-Based Sentiment Analysis & Interactive Web Studio

This document provides a comprehensive, presentation-ready walkthrough of **Project 2** for course defense, grading, and presentation.

---

## 1. The 4-Generation Model Evolution Trilogy

```
Gen 1: Baseline MLP              Gen 2: Imbalance Tuning           Gen 3: High-Quality Audit        Gen 4: Full-Corpus Champion
┌────────────────────────────┐   ┌─────────────────────────────┐   ┌────────────────────────────┐   ┌──────────────────────────────┐
│ [128, 64] Dense Layers     │   │ [128, 64] Dense Layers      │   │ [256, 128] Dense Layers    │   │ [384, 192] Dense Layers      │
│ 100 Initial Samples        │   │ 500 Samples + Damped Weights│   │ 1,000 Slide-Rule Audited   │   │ 2,000 Full-Corpus Cleaned    │
│ • Acc: 50.00%              │   │ • Acc: 76.35% (+26.35%)     │   │ • Acc: 85.31% (+8.96%)     │   │ • Acc: 87.64% (+2.33%)       │
│ • Mention F1: 0.416        │   │ • Mention F1: 0.626         │   │ • Mention F1: 0.714        │   │ • Mention F1: 0.790          │
│ • Macro F1: 0.284          │   │ • Macro F1: 0.394           │   │ • Macro F1: 0.497          │   │ • Macro F1: 0.554            │
│ • Loss: 1.351              │   │ • Loss: 0.684               │   │ • Loss: 0.492              │   │ • Loss: 0.399 (Deep Conv.)   │
└────────────────────────────┘   └─────────────────────────────┘   └────────────────────────────┘   └──────────────────────────────┘
```

---

## 2. Key Architecture & Technical Innovations

1. **Multi-Task Aspect-Sentiment Output**:
   - Instead of 8 independent binary models, a unified PyTorch MLP outputs $40$ logits ($8 \text{ aspects} \times 5 \text{ states}$), capturing inter-aspect correlations with only $180,328$ parameters.
2. **Solving Class Imbalance (Slide 27)**:
   - State 0 (*absent*) constitutes $>85\%$ of all slots. A naive model achieves $>80\%$ accuracy by predicting zeros everywhere.
   - We implemented a **square-root damped inverse-frequency class-weight schedule** with a class-0 cap ($w_0 \le 0.25$) and per-aspect frequency adjustments, forcing the network to actively learn positive/negative sentiment evidence.
3. **Rigorous Data Quality Audit**:
   - Filtered out 217 hotel management auto-replies and meaningless fragments.
   - Clarified aspect boundaries: equipment defects belong to `facilities`, air conditioning rattle belongs to `sound_insulation_noise`, and overall frustration ("免费我也不住了") does not invent an unmentioned `value` complaint.

---

## 3. Web Studio Demo Flow (Port 8792)

1. **Review-Level Sentiment Aggregation**:
   - Select any of the 30 independent demo reviews.
   - Watch the server split Chinese sentences, run Qwen 256-dim embeddings, and calculate aggregate aspect scores $S_a = 100 \times \frac{\sum s_{ia}}{n_a}$ on $[-100, +100]$.
2. **Single-Sentence Probe**:
   - Type custom sentences like `房间很干净，但是隔音太差了。`
   - Observe live multi-class probabilities across all 8 aspects simultaneously.
3. **Training Data Studio & 1-Click Retraining**:
   - Inspect all 2,000 records, filter model-disagreement samples, edit ground-truth states in the modal, and trigger background retraining with instant hot reload.
