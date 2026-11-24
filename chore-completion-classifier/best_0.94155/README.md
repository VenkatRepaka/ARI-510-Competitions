# Backup: submission that scored 0.94155 on the public leaderboard

Model: `all backbones | full | extra trees` - ExtraTrees (300 trees, sqrt features, balanced, random_state=0)
on frozen image embeddings from 5 backbones (dinov2_l, eva02_l, siglip_so400m, vith14_laion, vitl14_openai),
"full" view = mean of image + mirror embedding; probabilities averaged over the 5 backbones.
Trained on the 66 original photos only (no web images).

Files
- `train.py`, `extract_features.py`: frozen copies of the code (2026-11-15). Verified to reproduce
  `submission_public0.94155_extratrees.csv` exactly.
- `train_original.csv`: the 66 original training rows.
- Features: `../features_backup_before_external/` holds the exact feature files used
  (`../features/` has the same rows plus web photos appended, which this command ignores).

Reproduce (from the chore-completion-classifier folder; if the main train.py has since changed,
copy this folder's train.py and extract_features.py back over it first - they read data/ and features/
relative to their own location):

```bash
TRAIN_CSV=train_original.csv ../.venv/bin/python train.py --repeats 1 --modes full --entry "all backbones | full | extra trees" --out submission_repro.csv
```

Versions: scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1
