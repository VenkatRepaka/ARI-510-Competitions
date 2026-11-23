"""Train the chosen model on ALL labelled images and write the Kaggle submission file for the 236 test images.

WHAT THIS FILE DOES
    1. takes a model name and feature files from the command line (the same names baseline.py uses)
    2. trains that model on train + validation together (941 labelled images)
    3. predicts P(artifact) for the 236 test images and turns it into 0 / 1 (1 if P >= 0.5)
    4. checks the result has the exact rows and order of data/sample_submission.csv
    5. saves the CSV: two columns, image_id and artifact_present, 236 rows plus a header

WHY TRAIN ON TRAIN + VALIDATION HERE
    In baseline.py the validation images were held back so models could be compared fairly. Once a model has
    been chosen (by its CV score), holding images back no longer helps: training on all 941 instead of 823 gives
    the model 14% more examples to learn from. The settings stay exactly as they were when the model was scored.
    The test images are never used for training or for choosing anything; they are only predicted at the end.

WHY 0 / 1 AND NOT PROBABILITIES
    The competition asks for "0 or 1, not a probability", scored with F1 on class 1. The threshold 0.5 was
    checked in baseline.py: the best threshold on the out-of-fold predictions was 0.41-0.48 for the good models,
    close to 0.5, so 0.5 is used (choosing it on the test set is not possible and would not be allowed).

THE SUBMITTED FILES
    python make_submission.py svm_rbf dinov2_l siglip_so400m eva02_l quality --out submission_run1_all3q_svm.csv
        run1: three networks + 74 quality numbers, RBF SVM (dev F1 0.906, CV F1 0.886); 133 of 236 predicted 1
    python make_submission.py mlp_strong eva02_l quality --out submission_run2_eva02q_mlp.csv
        run2: EVA02 + quality numbers, MLP ensemble (dev F1 0.873, CV F1 0.880); 125 of 236 predicted 1
    Every random choice has a fixed seed, so each command writes exactly the same file every time.

USAGE
    python make_submission.py <model> <feature> [<feature> ...] [--thr 0.5] [--out name.csv]
      <model>     a name from baseline.models(): logreg, svm_rbf, hgb, mlp, mlp_h64, mlp_strong
      <feature>   one or more feature files from features/ (without ".npz"), joined side by side
      --thr       threshold for turning probabilities into 1 (default 0.5)
      --out       output file name (default: submission_<model>_<feature+feature...>.csv)

THE FEATURE NAMES = FILES SAVED BY THE EARLIER SCRIPTS
    Each <feature> word is the name of a file in features/ that an earlier script computed and saved.
    Nothing is computed from the images here; the script only loads these saved numbers
    (load_feature("quality") opens features/quality.npz).
        name            file                          numbers per image   saved by
        quality         features/quality.npz           74                 quality_features.py (hand-made quality numbers)
        dinov2_l        features/dinov2_l.npz          2048               embed_features.py   (DINOv2 embedding)
        siglip_so400m   features/siglip_so400m.npz     1152               embed_features.py   (SigLIP embedding)
        eva02_l         features/eva02_l.npz            768               embed_features.py   (EVA02 embedding)
    So those scripts must have been run first (once; about 5 + 30 minutes). The files are already in features/.
    run1 uses all four (2048 + 1152 + 768 + 74 = 4042 numbers per image); run2 uses eva02_l + quality (842).
"""
import argparse                 # reads the command-line arguments
from pathlib import Path        # file paths

import numpy as np
import pandas as pd

# Reuse baseline.py's functions, so the model and features are built EXACTLY as when they were scored.
from baseline import load_feature, load_labels, matrix, models

ROOT = Path(__file__).parent

# ----------------------------------------------------------------------------------------------------------
# 1. Read the command line
# ----------------------------------------------------------------------------------------------------------
# Example: python make_submission.py svm_rbf dinov2_l quality --out x.csv
#   -> a.model = "svm_rbf", a.features = ["dinov2_l", "quality"], a.thr = 0.5, a.out = "x.csv"
ap = argparse.ArgumentParser()
ap.add_argument("model")                                  # first word: the model name
ap.add_argument("features", nargs="+")                    # all following words: feature files (at least one)
ap.add_argument("--thr", type=float, default=0.5)         # optional threshold
ap.add_argument("--out")                                  # optional output file name
a = ap.parse_args()

# ----------------------------------------------------------------------------------------------------------
# 2. Train on all 941 labelled images
# ----------------------------------------------------------------------------------------------------------
tr, va = load_labels()                                    # train.csv (823) and validation.csv (118)
te = pd.read_csv(ROOT / "data/test.csv")                  # test.csv (236): image_id and filename, no labels
feats = [load_feature(n) for n in a.features]             # each a dictionary {image_id: vector}
ids = np.concatenate([tr.image_id, va.image_id])          # the 941 labelled image ids
y = np.concatenate([tr.artifact_present, va.artifact_present])   # their 941 labels (0 / 1)
# models()[a.model]() builds a fresh, untrained model with the same settings as in baseline.py;
# .fit trains it on the 941 feature rows and labels.
clf = models()[a.model]().fit(matrix(feats, ids), y)

# ----------------------------------------------------------------------------------------------------------
# 3. Predict the test images
# ----------------------------------------------------------------------------------------------------------
# Feature rows for the test images in test.csv order; [:, 1] = P(artifact) for each.
p = clf.predict_proba(matrix(feats, te.image_id))[:, 1]
# 1 if the probability reaches the threshold, else 0 (astype(int) turns True / False into 1 / 0).
sub = pd.DataFrame({"image_id": te.image_id, "artifact_present": (p >= a.thr).astype(int)})

# ----------------------------------------------------------------------------------------------------------
# 4. Check, then save
# ----------------------------------------------------------------------------------------------------------
# The host's template: the same 236 image_ids in the required order. The assert stops the script with an error
# (and writes nothing) if the ids or their order differ, or if there are not exactly 236 rows.
sample = pd.read_csv(ROOT / "data/sample_submission.csv")
assert list(sub.image_id) == list(sample.image_id) and len(sub) == 236
# Output name: --out if given, otherwise built from the model and features,
# e.g. submission_svm_rbf_dinov2_l+quality.csv
out = a.out or f"submission_{a.model}_{'+'.join(a.features)}.csv"
sub.to_csv(ROOT / out, index=False)                       # index=False: no extra row-number column
# e.g. "wrote submission_run1_all3q_svm.csv: 133 of 236 predicted artifact (56%)". The share of 1s is a quick
# sanity check: the labelled data is 54% artifact, so a share far from that would suggest something is wrong.
print(f"wrote {out}: {sub.artifact_present.sum()} of {len(sub)} predicted artifact ({sub.artifact_present.mean():.0%})")
