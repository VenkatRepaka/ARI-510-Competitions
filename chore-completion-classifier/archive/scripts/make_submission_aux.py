"""Submission files for the CDCM overflow-score models (chosen by 10x5-fold CV in aux_cv.py).

  submission_dinov2_tiles_svm.csv    : DINOv2 full (image + mirror) + 4 tile scores (overflow_tiles.npz)  + RBF SVM  (CV 0.890)
  submission_dinov2_overflow_svm.csv : DINOv2 full (image + mirror) + 4 whole-image overflow scores (overflow_score.npz) + RBF SVM  (CV 0.875)
  submission_dinov2_tiles_open_svm.csv : as the tiles file + 2 open-bag scores x4 (bag_tiles.npz open_tile_max/top2),
                                         66 + 19 external photos  (aux_cv.py --open, CV 0.912 +/- 0.020: best)
  submission_dinov2_open_routed.csv  : the tiles+open SVM decides the chore and the trash label; photos it calls dishes
                                       get Complete/Incomplete from a dishes-only logistic regression (C=1, balanced)
                                       on DINOv2 full + 10 augmented views per training photo (dishes_cv.py, CV 0.916)

Training set: every photo in data/train.csv (66 competition photos + external images), same features and
model settings as in aux_cv.py (SVC rbf, C=3, gamma='scale', class_weight='balanced'; ALPHA = 1).
The RBF SVM is deterministic, so rerunning gives identical files.

Prerequisites (in this order): extract_features.py, overflow_aux.py (encode, score), overflow_aux.py tiles.
Usage: python make_submission_aux.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from extract_features import CLASSES, unit

ROOT = Path(__file__).parent
ALPHA = 1.0
train = pd.read_csv(ROOT / "data/train.csv")
test = pd.read_csv(ROOT / "data/test.csv")


def dinov2_full(ids):
    z = np.load(ROOT / "features/dinov2_l.npz")
    pos = {i: k for k, i in enumerate(z["ids"])}
    idx = [pos[i] for i in ids]
    return unit(z["img"][idx] + z["img_flip"][idx])


def scores(path, keys, ids):
    z = np.load(ROOT / path)
    pos = {i: k for k, i in enumerate(z["ids"])}
    return np.stack([z[k][[pos[i] for i in ids]] for k in keys], 1)


EXTRA = {
    "submission_dinov2_tiles_svm.csv": ("features/overflow_tiles.npz",
        ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]),
    "submission_dinov2_overflow_svm.csv": ("features/overflow_score.npz",
        ["dinov2_l_full", "dinov2_l_max7", "siglip_so400m_full", "siglip_so400m_max7"]),
}


def main():
    y = train.label.map(CLASSES.index).values
    for out, (path, keys) in EXTRA.items():
        Xtr = np.hstack([dinov2_full(train.id), ALPHA * scores(path, keys, train.id)])
        Xte = np.hstack([dinov2_full(test.id), ALPHA * scores(path, keys, test.id)])
        clf = SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced").fit(Xtr, y)
        sub = pd.DataFrame({"id": test.id, "label": [CLASSES[k] for k in clf.predict(Xte)]})
        sub.to_csv(ROOT / out, index=False)
        print(f"{out}: trained on {len(train)} photos, {len(sub)} test predictions; "
              f"label counts {sub.label.value_counts().to_dict()}")



TILE_KEYS = EXTRA["submission_dinov2_tiles_svm.csv"][1]


def main_open():
    from sklearn.linear_model import LogisticRegression
    y = train.label.map(CLASSES.index).values
    feats = lambda ids: np.hstack([dinov2_full(ids), ALPHA * scores("features/overflow_tiles.npz", TILE_KEYS, ids),
                                   ALPHA * 4.0 * scores("features/bag_tiles.npz", ["open_tile_max", "open_tile_top2"], ids)])
    svm = SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced").fit(feats(train.id), y)
    pred = svm.predict(feats(test.id))
    out = {"submission_dinov2_tiles_open_svm.csv": pred.copy()}

    # routed: dishes-only LR on DINOv2 full + 10 augmented views (features/dinov2_l_aug.npz)
    d = np.isin(y, [0, 1]); d_ids = train.id.values[d]
    z = np.load(ROOT / "features/dinov2_l_aug.npz"); pos = {i: k for k, i in enumerate(z["ids"])}
    aug = unit(z["aug"][[pos[i] for i in d_ids]])[:, :10]
    Xd = dinov2_full(d_ids); yd = y[d]
    lr = LogisticRegression(C=1, max_iter=5000, class_weight="balanced").fit(
        np.vstack([Xd, aug.reshape(-1, Xd.shape[1])]), np.r_[yd, np.repeat(yd, 10)])
    routed = pred.copy(); td = np.isin(pred, [0, 1])
    routed[td] = lr.predict(dinov2_full(test.id.values[td]))
    out["submission_dinov2_open_routed.csv"] = routed
    for name, pr in out.items():
        sub = pd.DataFrame({"id": test.id, "label": [CLASSES[k] for k in pr]})
        sub.to_csv(ROOT / name, index=False)
        print(f"{name}: {len(sub)} predictions; {sub.label.value_counts().to_dict()}")


if __name__ == "__main__":
    import sys
    main_open() if "--open" in sys.argv else main()
