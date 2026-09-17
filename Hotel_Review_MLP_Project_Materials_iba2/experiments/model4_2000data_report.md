# Model Gen 4 (2000-Data Full-Corpus Champion) Report

**Date**: 2026-09-12  
**Checkpoint**: `experiments/run_model4_champion_2000v4/best_model.pt`  
**Data**: `labels_final_2000_v4_mlp.jsonl` — 2000 rows, **1783 trainable**, 217 excluded, **3053** supervised mentions  
**Split**: Group by `review_id` (train / val / test ≈ 70% / 15% / 15%; held-out test ≈ 267)  
**Architecture**: HotelReviewMLP `256 → 384 → 192 → 40` (180,328 params)  
**Config**: batch=16, dropout=0.25, lr=1e-3, class0_weight=0.25, seed=42, early-stop best epoch **63**

---

## 1. Four-generation comparison

| Metric | Gen 1 (100) | Gen 2 (500) | Gen 3 (1000) | **Gen 4 (2000 v4)** | Delta G1→G4 |
| --- | :---: | :---: | :---: | :---: | :---: |
| Trainable samples | 89 | 435 | 879 | **1783** | +1903% |
| Overall Accuracy | 50.00% | 76.35% | 85.31% | **87.64%** | **+37.64%** |
| Mention Precision | 31.37% | 54.71% | 66.80% | **74.57%** | +43.20% |
| Mention Recall | 61.54% | 73.23% | 76.74% | **84.00%** | +22.46% |
| **Mention F1** | 0.416 | 0.626 | 0.714 | **0.790** | **+89.9%** |
| **Macro F1** | 0.284 | 0.394 | 0.497 | **0.554** | **+95.1%** |
| **Non-Absent Macro F1** | 0.120 | 0.280 | 0.391 | **0.458** | **~3.8×** |
| Test Loss | 1.351 | 0.684 | 0.492 | **0.399** | **-70.5%** |

---

## 2. Per-aspect (Gen 4 test set)

| Aspect | Accuracy | Mention F1 |
| --- | :---: | :---: |
| cleanliness | 89.9% | 0.741 |
| service | 84.1% | 0.823 |
| location | 91.5% | 0.870 |
| facilities | 78.8% | 0.814 |
| room_comfort | 76.5% | 0.648 |
| sound_insulation_noise | 97.3% | 0.928 |
| food | 92.4% | 0.795 |
| value | 81.4% | 0.741 |

---

## 3. Hyperparameter search (same seed / labels)

| Config | Acc | Mention F1 | Macro F1 | Non-Abs F1 |
| --- | :---: | :---: | :---: | :---: |
| **[384, 192] bs16 dp0.25** | **87.64%** | **0.790** | **0.554** | **0.458** |
| [384, 192] bs16 dp0.20 | 86.89% | 0.779 | 0.559 | 0.465 |
| [384, 192] dp0.25 c0=0.20 | 86.52% | 0.774 | 0.548 | 0.453 |
| [256, 128] bs16 dp0.20 | 85.07% | 0.756 | 0.516 | 0.414 |
| [256, 128] (v3 champion arch) | 84.55% | 0.742 | 0.523 | 0.423 |
| [256, 128, 64] | 80.48% | 0.690 | 0.478 | 0.375 |

**Conclusion**: With 2× more high-quality labels, wider `[384, 192]` capacity helps; deeper 3-layer nets still overfit.

---

## 4. Deployment

- Web JS weights: `mlp_weights.js` / `mlp_weights.json` (exported from Gen 4 checkpoint)
- Backend auto-loads `experiments/run_model4_champion_2000v4`
- UI banner + Gen1–4 comparison tables updated in `index.html` / `app.js`
- Start: `python server.py` → http://localhost:8792
