# MLP Optimization Analysis (Slide-Compliant) — Gen 4

Date: 2026-09-12  
Baseline: `v4_arch384_192` / `run_model4_champion_2000v4`  
Data: `labels_final_2000_v4_mlp.jsonl` (1783 trainable)  
**Fixed evaluation protocol**: Group split by `review_id`, **split-seed = 42** (same test set for all comparisons).

---

## 1. What the slides already lock in

| Constraint | Status |
| --- | --- |
| Classifier is MLP only | Kept |
| Input = fixed 256-d Qwen embedding (no encoder fine-tune) | Kept |
| Output 40 logits → 8×5, one softmax per aspect | Kept |
| CrossEntropyLoss on logits | Kept (weighted / smoothed CE still CE) |
| Group split by `review_id` | Kept + **split seed now independent of train seed** |
| Report Mention P/R/F1, Macro/Micro F1, per-aspect, errors | Kept |

---

## 2. Baseline (production champion)

| Metric | Score |
| --- | :---: |
| Overall Accuracy | **87.64%** |
| Mention F1 | **0.790** |
| Macro F1 | **0.554** |
| Non-Absent Macro F1 | **0.458** |
| Arch | 256→384→192→40 (180,328 params) |

Weakest heads: `room_comfort` (F1≈0.68), `food` (≈0.70), `cleanliness` (≈0.73) — driven by 11–14% mention rates and almost no neutral/mixed examples.

---

## 3. Experiments run (all on same split-seed 42)

| Run | Acc | Mention F1 | Macro F1 | Non-Abs | Verdict |
| --- | :---: | :---: | :---: | :---: | --- |
| **Champion [384,192] dp0.25** | **87.64** | **0.790** | **0.554** | **0.458** | Best balanced |
| BatchNorm + per-aspect CE + LS0.05 | 86.05 | 0.782 | 0.518 | 0.414 | Worse (BN over-regularized) |
| Per-aspect CE + LS0.03 | 86.89 | 0.791 | 0.507 | 0.399 | Mention ok, macro worse |
| Label smoothing 0.05 only | 85.72 | 0.779 | 0.511 | 0.406 | Worse |
| class0_weight=0.20 | 87.73 | 0.790 | 0.535 | 0.435 | Acc ≈, macro worse |
| [448,224] c0=0.22 | 87.92 | 0.789 | 0.525 | 0.421 | Acc slight ↑, macro ↓ |
| Fair seed 45 (split=42) | 87.97 | 0.791 | 0.546 | 0.448 | Good, still ≤ champion on macro |
| Fair seed 46 (split=42) | **88.53** | **0.802** | 0.538 | 0.436 | Best Acc/Mention, weaker macro |
| Ensemble champion ⊕ seed45 (logit avg) | 88.20 | 0.799 | 0.550 | 0.452 | Tiny Acc/Mention gain |

### Critical methodological fix
Earlier “seed44 = 93% Acc” was **invalid**: `--seed` also changed the GroupKFold split, so train reviews leaked into another seed’s test set (194 overlapping reviews).  
Added `--split-seed` so training seeds can be compared **without leakage**.

---

## 4. Code improvements delivered in `train_mlp.py`

1. `--split-seed` — decouple review_id split from init/shuffle seed  
2. `--per-aspect-weighting` — (8,5) CE weights (helps sparse food/sound heads in theory)  
3. `--label-smoothing` — still CrossEntropyLoss  
4. `--rare-class-boost` — extra weight on rare states 2/4  
5. BatchNorm-safe `drop_last` training loader  

---

## 5. What still looks promising (not yet fully mined)

1. **More clean labels** (biggest lever historically: 89→435→879→1783 each beat architecture tweaks). Human spot-fix of `room_comfort` / `food` / mixed states would likely beat more hyperparams.  
2. **Sample-level reweighting** — up-weight sentences that contain rare aspects during CE (still CE).  
3. **3–5 model logit ensemble** with `--split-seed 42` + different `--seed` — expect +0.3–0.6 Acc, +0.01 Mention F1 if you accept ~3× inference cost.  
4. **Keep champion for Macro/Non-Absent**; use seed46 only if classroom demo cares most about overall accuracy.  
5. Do **not** spend more time on BN / heavy label-smoothing / much deeper nets — they regressed here.

---

## 6. Recommendation

- **Default / web deploy**: keep `run_model4_champion_2000v4` (best Macro + Non-Absent F1).  
- **Demo “higher accuracy” mode**: `v4_fair_seed46` (88.53% Acc, 0.802 Mention F1).  
- **Optional ensemble**: champion ⊕ seed45 (88.20% / 0.799) if you want a small extra bump.  
- Next real jump = label quality on weak aspects, not another architecture lottery.
