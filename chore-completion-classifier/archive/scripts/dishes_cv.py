"""Dishes-only comparison: Complete vs Incomplete on the 33 original dishes photos.

Chore type is ~perfect, so dishes can get their own model (routing). Same 10x5 photo-grouped splits as aux_cv.py
(drawn over all 66 originals); only the dishes photos of each split are used. External #912 (clean sink) is
always in training. Reports dishes macro F1 (mean over 10 repeats) and how often each photo is wrong.

Grid: backbone features (full = image + mirror, mean7 = mean of 7 views) x model x augmentation (+20 aug views per
training photo, where features/<b>_aug.npz exists).
Pseudo-labels (--pseudo): the 36 unlabelled test photos; those the 4-class SVM calls dishes are labelled by the
current dishes model and the confident ones (|p - 0.5| >= margin) are added to training. No test label is used.
-> logs/dishes_cv.csv, logs/dishes_cv_photos.csv.  Usage: python dishes_cv.py [--pseudo]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import aux_cv as A
from extract_features import unit

ROOT = A.ROOT
test = pd.read_csv(ROOT / "data/test.csv")
B5 = ["dinov2_l", "eva02_l", "siglip_so400m", "vith14_laion", "vitl14_openai"]


def load(b, ids):
    z = np.load(ROOT / f"features/{b}.npz"); p = {int(i): k for k, i in enumerate(z["ids"])}; ix = [p[i] for i in ids]
    full = unit(z["img"][ix] + z["img_flip"][ix])
    mean7 = unit(np.concatenate([z["img"][ix][:, None], z["img_flip"][ix][:, None], z["crops"][ix]], 1).mean(1))
    return full, mean7


def load_aug(b, ids):
    f = ROOT / f"features/{b}_aug.npz"
    if not f.exists():
        return None
    z = np.load(f); p = {int(i): k for k, i in enumerate(z["ids"])}
    if not all(i in p for i in ids):
        return None
    return unit(z["aug"][[p[i] for i in ids]])                          # (N, 20, D)


train = A.train
dish_rows = np.where(train.label.str.startswith("Washing").values)[0]
d_ids = list(train.id.values[dish_rows])
yd = train.label.values[dish_rows]
yd = np.array([1 if s.endswith("Incomplete") else 0 for s in yd])     # 1 = Incomplete
is_orig = np.isin(dish_rows, A.orig)
t_ids = list(test.id)

FEATS, AUGS = {}, {}
for b in B5:
    for kind, Xa, Xt in zip(("full", "mean7"), load(b, d_ids), load(b, t_ids)):
        FEATS[f"{b} {kind}"] = (Xa, Xt)
    AUGS[b] = load_aug(b, d_ids)
cat = lambda k: tuple(unit(np.hstack([FEATS[f"{b} {k}"][s] for b in B5])) for s in (0, 1))
FEATS["concat5 full"], FEATS["concat5 mean7"] = cat("full"), cat("mean7")
FEATS["dinov2+siglip full"] = tuple(unit(np.hstack([FEATS[f"{b} full"][s] for b in ("dinov2_l", "siglip_so400m")])) for s in (0, 1))

BAL = dict(class_weight="balanced")
MODELS = {"LR C=1": lambda: LogisticRegression(C=1, max_iter=5000, **BAL),
          "LR C=0.1": lambda: LogisticRegression(C=0.1, max_iter=5000, **BAL),
          "RBF SVM": lambda: SVC(kernel="rbf", C=3, gamma="scale", probability=False, **BAL),
          "kNN 5": lambda: KNeighborsClassifier(5, metric="cosine", weights="distance"),
          "extra trees": lambda: ExtraTreesClassifier(300, max_features="sqrt", n_jobs=-1, random_state=0, **BAL)}


def score(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-clf.decision_function(X)))


# splits: same as aux_cv (over 66 originals, 4-class stratified); map to dishes rows
splits66 = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(A.orig, A.y[A.orig]))
pos = {r: k for k, r in enumerate(dish_rows)}
SPLITS = []
for tr, va in splits66:
    va_d = [pos[r] for r in A.orig[va] if r in pos]
    tr_d = [pos[r] for r in A.orig[tr] if r in pos] + [k for k in range(len(dish_rows)) if not is_orig[k]]
    SPLITS.append((np.array(tr_d), np.array(va_d)))
ev = np.where(is_orig)[0]

# test photos the 4-class base model calls dishes (for pseudo-labelling)
X4 = A.BASE["dinov2 full"]
svm4 = A.MODELS["RBF SVM"]().fit(X4, A.y)
Xt4 = load("dinov2_l", t_ids)[0]
t_dish = np.isin(svm4.predict(Xt4), [0, 1])


def run(feat, model, aug=0, pseudo=None):
    Xa, Xt = FEATS[feat]
    b = feat.split()[0]
    P = np.full((A.REPEATS, len(yd)), np.nan)
    for k, (tr, va) in enumerate(SPLITS):
        Xtr, ytr = Xa[tr], yd[tr]
        if aug and AUGS.get(b) is not None:
            Xtr = np.vstack([Xtr, AUGS[b][tr, :aug].reshape(-1, Xa.shape[1])]); ytr = np.r_[ytr, np.repeat(yd[tr], aug)]
        clf = MODELS[model]().fit(Xtr, ytr)
        if pseudo is not None:
            pt = score(clf, Xt[t_dish])
            keep = np.abs(pt - 0.5) >= pseudo
            clf = MODELS[model]().fit(np.vstack([Xtr, Xt[t_dish][keep]]), np.r_[ytr, (pt[keep] > .5).astype(int)])
        P[k // 5, va] = score(clf, Xa[va])
    pred = (P[:, ev] > .5).astype(int)
    f1 = [f1_score(yd[ev], p, average="macro") for p in pred]
    wrong = {int(d_ids[j]): int(n) for j, n in zip(ev, (pred != yd[ev]).sum(0)) if n}
    return np.mean(f1), np.std(f1), wrong


def main():
    rows = []
    pseudo = "--pseudo" in sys.argv
    grid = [(f, m, a) for f in FEATS for m in MODELS for a in (0, 10)
            if not (a and (f.split()[0] not in AUGS or AUGS[f.split()[0]] is None or "mean7" in f))]
    for f, m, a in grid:
        for ps in ([None, 0.3, 0.2] if pseudo else [None]):
            mu, sd, wrong = run(f, m, a, ps)
            rows.append({"features": f, "model": m, "aug views": a, "pseudo margin": ps, "dishes F1": round(mu, 3),
                         "sd": round(sd, 3), "n wrong": sum(wrong.values()) / A.REPEATS, "wrong": wrong})
            print(f"{f:22s} {m:12s} aug {a:2d} pseudo {str(ps):4s}  F1 {mu:.3f} ± {sd:.3f}  wrong {wrong}", flush=True)
    df = pd.DataFrame(rows).sort_values("dishes F1", ascending=False)
    df.to_csv(ROOT / ("logs/dishes_cv_pseudo.csv" if pseudo else "logs/dishes_cv.csv"), index=False)
    print(f"\n{t_dish.sum()} of {len(t_ids)} test photos called dishes")
    print(df.head(15).drop(columns="wrong").to_string(index=False))


if __name__ == "__main__":
    main()
