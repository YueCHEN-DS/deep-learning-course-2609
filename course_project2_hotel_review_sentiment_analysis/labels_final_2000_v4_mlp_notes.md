# labels_final_2000_v4_mlp.jsonl — final MLP training set

Highest-quality labels after:
1. DeepSeek labeling (full 2,000)
2. Slide-rule multi-pass audits
3. Spot-audit report (`labels_final_1000_spot_audit_report.md`)
4. Batch-10 deep review + systematic sweeps

## Counts
| field | value |
| --- | --- |
| rows | 2000 |
| trainable (`human_confirmed`) | **1783** |
| excluded | 217 |
| needs_review | 0 |
| multi-aspect (train) | 832 |
| all-absent (train) | 80 |

## Aspect mentions (train)
| aspect | count |
| --- | --- |
| cleanliness | 245 |
| service | 663 |
| location | 382 |
| facilities | 561 |
| room_comfort | 360 |
| sound_insulation_noise | 205 |
| food | 200 |
| value | 437 |

## v4 policies applied
- Generic praise (`还不错` / `总体还可以`) does **not** create a value label
- Nuisance phone calls → **sound**, not service
- `一般` / `一般般` → **neutral**
- Smell/smoke (`烟味`/`异味`) → cleanliness, not sound
- Hotel replies / third-party restaurants / OTA-only complaints excluded or all-absent
- Evidence must be an exact substring of the target sentence

## Train
```python
import json
rows = [json.loads(l) for l in open('labels_final_2000_v4_mlp.jsonl')]
train = [r for r in rows if r['trainable']]  # 1783
# join hotel_embeddings_256.npz by sentence_id
# GroupKFold / split by review_id
```

Aspect order: cleanliness, service, location, facilities, room_comfort, sound_insulation_noise, food, value  
States: 0 absent, 1 neg, 2 neu, 3 pos, 4 mixed
