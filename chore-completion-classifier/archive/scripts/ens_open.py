"""Ensembles of the current models (all with DINOv2 full + 4 overflow tile scores + 2 open-bag scores, train.csv 66 + 19 ext).
Components (out-of-fold probabilities, same 10x5 photo-grouped CV as aux_cv.py, dev fold = original photos only):
  svm      : RBF SVM, no augmentation                                  (CV 0.912 alone)
  lr_rand  : logistic regression + 30 random views per photo (rand_aug)  (0.913)
  lr_zoom  : logistic regression + zoom in + out (zoom_aug)              (0.905)
  mlp_rand : neural net + 10 random views                                (0.862)
Augmented rows copy their photo's open-bag score. Probabilities are averaged with equal weights.
Pre-chosen main result: svm + lr_rand. Other combinations are for reference (picking the best is optimistic).
  -> logs/ens_open_cv.csv, logs/ens_open_cv_photos.csv
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from aux_cv import ALPHA, BASE, MODELS, REPEATS, ROOT, ext, orig, proba, train, y
from extract_features import OUT, unit

t = np.load(ROOT / "features/overflow_tiles.npz"); tpos = {i: k for k, i in enumerate(t["ids"])}
TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
TILE = ALPHA * np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1)
b = np.load(ROOT / "features/bag_tiles.npz"); bpos = {i: k for k, i in enumerate(b["ids"])}
O = ALPHA * 4.0 * np.stack([b[k][[bpos[i] for i in train.id]] for k in ("open_tile_max", "open_tile_top2")], 1)
X = np.hstack([BASE["dinov2 full"], TILE, O])
r = np.load(OUT / "rand_aug_dinov2_l.npz"); assert list(r["ids"]) == list(train.id)
R = np.concatenate([unit(r["img"] + r["mirror"]), np.repeat(np.hstack([TILE, O])[:, None], r["img"].shape[1], 1)], -1)
z = np.load(OUT / "zoom_aug.npz"); assert list(z["ids"]) == list(train.id)
Z = np.concatenate([z["dinov2"], ALPHA * z["tiles"], np.repeat(O[:, None], z["dinov2"].shape[1], 1)], -1)


def aug_rows(A, n, tr):
    return np.vstack([X[tr], A[tr, :n].reshape(-1, A.shape[-1])]), np.concatenate([y[tr], np.repeat(y[tr], n)])


COMP = {"svm": ("RBF SVM", None, 0), "lr_rand": ("logistic regression", R, 30),
        "lr_zoom": ("logistic regression", Z, Z.shape[1]), "mlp_rand": ("neural net (MLP)", R, 10)}
splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
          RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
P = {c: np.zeros((REPEATS, len(train), 4)) for c in COMP}
for c, (m, A, n) in COMP.items():
    for s, (tr, va) in enumerate(splits):
        Xtr, ytr = (X[tr], y[tr]) if A is None else aug_rows(A, n, tr)
        P[c][s // 5, va] = proba(MODELS[m]().fit(Xtr, ytr), X[va])
    print(f"{c} done", flush=True)

watch = [72, 19, 13, 63, 77, 41, 26, 2, 60, 31, 101, 12]
rows, photos = [], {}
for k in range(1, len(COMP) + 1):
    for combo in combinations(COMP, k):
        Pm = np.mean([P[c] for c in combo], 0)
        f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in Pm]
        name = " + ".join(combo)
        rows.append({"ensemble": name, "n": k, "CV macro F1": np.mean(f1), "std": np.std(f1)})
        photos[name] = {f"#{w}": f"{(Pm[:, j].argmax(1) != y[j]).sum()}/{REPEATS}"
                        for w in watch for j in [int(np.where(train.id == w)[0][0])]}
res = pd.DataFrame(rows).sort_values("CV macro F1", ascending=False)
res.to_csv(ROOT / "logs/ens_open_cv.csv", index=False)
pd.DataFrame(photos).to_csv(ROOT / "logs/ens_open_cv_photos.csv")
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
print("\nTimes wrong out of 10:")
print(pd.DataFrame(photos)[list(res.ensemble[:6]) + ["svm", "lr_rand"]].drop_duplicates().T.to_string())
