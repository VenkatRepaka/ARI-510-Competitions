"""Learned open-vs-tied bag helper: a small classifier replaces the hand-made open-bag score.

The hand-made open-bag score (bag_tiles.py) compares each tile with the AVERAGE open-bag photo and the AVERAGE
tied-bag photo. Here the same knowledge is LEARNED instead, the same way the overflow tiles are learned from CDCM:

  helper training data : the 12 licensed external bag photos already in the training set
                         - tied (label 0): 901, 902, 903, 904, 905, 907, 910
                         - open (label 1): 951, 952, 953, 954, 955
                         rows per photo: the full photo and its mirror + its 13 tiles (2x2 + 3x3 grid), all DINOv2,
                         each row with the photo's label
  helper               : logistic regression (C=1, balanced), like the CDCM overflow helper in overflow_aux.py
  applied to           : the 13 tiles of every photo (features/dinov2_l_tiles.npz)
  new inputs           : bag_open_max (most open-looking tile), bag_open_top2 (mean of the two highest);
                         they are probabilities in [0, 1] like the overflow tile scores, so no extra scaling
No competition photo or label is used to train the helper, so the CV of the chore model stays honest.

CV (same 10 x 5 photo-grouped splits as aux_cv.py, TRAIN_CSV default data/train_best_913inc.csv), RBF SVM and LR:
  DINOv2 + tiles                        (no bag input)
  DINOv2 + tiles + hand-made open score (current best, x4 scaling)
  DINOv2 + tiles + learned bag score    (this script)
-> features/bag_learned.npz, logs/learned_bag_cv.csv, logs/learned_bag_photos.csv
Usage: python learned_bag.py
"""
import os

os.environ.setdefault("TRAIN_CSV", "data/train_best_913inc.csv")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import crop_top_cv as C
from extract_features import OUT, unit

A = C.A
y, orig, ext, ids = C.y, C.orig, C.ext, C.ids
TIED, OPEN = [901, 902, 903, 904, 905, 907, 910], [951, 952, 953, 954, 955]


def helper_scores():
    z = np.load(OUT / "dinov2_l.npz"); zp = {int(i): k for k, i in enumerate(z["ids"])}
    t = np.load(OUT / "dinov2_l_tiles.npz"); tp = {int(i): k for k, i in enumerate(t["ids"])}
    Xh, yh = [], []
    for lab, group in ((0, TIED), (1, OPEN)):
        for i in group:
            rows = [unit(z["img"][zp[i]]), unit(z["img_flip"][zp[i]])] + list(t["tiles"][tp[i]])
            Xh += rows; yh += [lab] * len(rows)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(np.array(Xh), np.array(yh))
    allids = np.array(sorted(tp))
    p = clf.predict_proba(t["tiles"][[tp[i] for i in allids]].reshape(-1, t["tiles"].shape[-1]))[:, 1].reshape(len(allids), -1)
    out = dict(ids=allids, bag_open_tiles=p, bag_open_max=p.max(1), bag_open_top2=np.sort(p, 1)[:, -2:].mean(1))
    np.savez(OUT / "bag_learned.npz", **out)
    print(f"helper trained on {len(yh)} rows from {len(TIED)} tied + {len(OPEN)} open bag photos")
    return out


def main():
    h = helper_scores(); hp = {int(i): k for k, i in enumerate(h["ids"])}
    LEARN = np.stack([h[k][[hp[i] for i in ids]] for k in ("bag_open_max", "bag_open_top2")], 1)
    full, ex = A.BASE["dinov2 full"], C.extras()             # ex[:, :4] overflow tiles, ex[:, 4:] hand-made open x4
    trash = orig[y[orig] >= 2]
    for name, s in (("hand-made open score", ex[:, 4]), ("learned bag score", LEARN[:, 0])):
        print(f"{name:22s} alone, Trash Incomplete vs Complete on the 33 competition trash photos: "
              f"AUC {roc_auc_score(y[trash] == 3, s[trash]):.3f}")
    V = {"DINOv2 + tiles": np.hstack([full, ex[:, :4]]),
         "DINOv2 + tiles + hand-made open score": np.hstack([full, ex]),
         "DINOv2 + tiles + learned bag score": np.hstack([full, ex[:, :4], LEARN])}
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig])]
    dsh, trs = orig[y[orig] < 2], orig[y[orig] >= 2]
    rows, wrong = [], {}
    for v, X in V.items():
        for m in ("RBF SVM", "logistic regression"):
            P = np.zeros((A.REPEATS, len(y)), int)
            for k, (tr, va) in enumerate(splits):
                P[k // 5, va] = A.proba(A.MODELS[m]().fit(X[tr], y[tr]), X[va]).argmax(1)
            f = [f1_score(y[orig], p[orig], average="macro") for p in P]
            rows.append({"features": v, "model": m, "CV macro F1": np.mean(f), "std": np.std(f),
                         "dishes F1": np.mean([f1_score(y[dsh], p[dsh], labels=[0, 1], average="macro") for p in P]),
                         "trash F1": np.mean([f1_score(y[trs], p[trs], labels=[2, 3], average="macro") for p in P])})
            if m == "RBF SVM":
                wrong[v] = {f"#{ids[j]}": int((P[:, j] != y[j]).sum()) for j in orig}
    r = pd.DataFrame(rows); r.to_csv(A.ROOT / "logs/learned_bag_cv.csv", index=False)
    w = pd.DataFrame(wrong); w = w[(w > 0).any(axis=1)]; w.to_csv(A.ROOT / "logs/learned_bag_photos.csv")
    pd.set_option("display.width", 220)
    print("\n" + r.round(3).to_string(index=False))
    print("\nRBF SVM, photos wrong at least once (out of 10):\n" + w.to_string())


if __name__ == "__main__":
    main()
