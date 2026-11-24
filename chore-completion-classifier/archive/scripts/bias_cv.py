"""Per-chore "Incomplete" bias on top of the best model, chosen inside the training folds.

Why: the remaining dishes errors are mostly Incomplete photos called Complete (#60 plate in sink + pan left out,
#2 food bowl next to a clean sink, #31), because a small mess in a tidy scene gives only a weak signal.
A bias b added to the Incomplete class score (separately for dishes, b_d, and trash, b_t) shifts the decision
towards Incomplete (b > 0) or Complete (b < 0) without changing which chore is predicted.

Base model: DINOv2 + tiles + open-bag score, RBF SVM (as aux_cv.py --open), class scores = softmax of the
one-vs-rest decision function (aux_cv.proba). Training set: TRAIN_CSV (default data/train_newbatch_clean.csv).
b_d and b_t are picked from a grid (-1.0 .. 1.0, step 0.05) by an inner 5-fold CV on the training photos of each
outer fold (best macro F1; ties -> the b closest to 0), never on the photo being scored.
Outer CV: the same 10 x 5 photo-grouped splits as aux_cv.py; only the 66 competition photos are scored.
-> logs/bias_cv.csv, logs/bias_cv_photos.csv
Usage: python bias_cv.py
"""
import os

os.environ.setdefault("TRAIN_CSV", "data/train_newbatch_clean.csv")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import priority_rule as R

A, X, y, ids, orig, ext = R.A, R.X, R.y, R.ids, R.orig, R.ext
GRID = np.round(np.arange(-1.0, 1.0001, 0.05), 2)
WATCH = [60, 2, 63, 50, 7, 31, 26, 101, 12, 13, 28]


def scores(tr, va):
    return A.proba(A.MODELS["RBF SVM"]().fit(X[tr], y[tr]), X[va])


def decide(S, bd, bt):
    S = S.copy()
    S[:, 1] += bd          # Washing Dishes|Incomplete
    S[:, 3] += bt          # Taking Out Trash|Incomplete
    return S.argmax(1)


def pick(S, yt):
    """Best (b_d, b_t) on held-out training scores; b_d only moves dishes, b_t only trash, so pick them separately."""
    best = {}
    for name, col in (("bd", 1), ("bt", 3)):
        res = []
        for b in GRID:
            bd, bt = (b, 0.0) if name == "bd" else (0.0, b)
            res.append((round(f1_score(yt, decide(S, bd, bt), average="macro"), 6), -abs(b), b))
        best[name] = max(res)[2]
    return best["bd"], best["bt"]


def main():
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig])]
    V = ["base", "dishes bias only", "dishes + trash bias"]
    P = {v: np.zeros((A.REPEATS, len(y)), int) for v in V}
    B = []
    for k, (tr, va) in enumerate(splits):
        tro = tr[np.isin(tr, orig)]
        S_in = np.zeros((len(tro), 4))
        for itr, iva in StratifiedKFold(5, shuffle=True, random_state=k).split(tro, y[tro]):
            S_in[iva] = scores(np.r_[tro[itr], ext], tro[iva])
        bd, bt = pick(S_in, y[tro])
        B.append((bd, bt))
        S = scores(tr, va); r = k // 5
        P["base"][r, va] = S.argmax(1)
        P["dishes bias only"][r, va] = decide(S, bd, 0.0)
        P["dishes + trash bias"][r, va] = decide(S, bd, bt)
        print(f"fold {k + 1}/{len(splits)}  b_dishes {bd:+.2f}  b_trash {bt:+.2f}", flush=True)
    dsh, trs = orig[y[orig] < 2], orig[y[orig] >= 2]
    rows, wrong = [], {}
    for v in V:
        f = [f1_score(y[orig], p[orig], average="macro") for p in P[v]]
        fd = [f1_score(y[dsh], p[dsh], labels=[0, 1], average="macro") for p in P[v]]
        ft = [f1_score(y[trs], p[trs], labels=[2, 3], average="macro") for p in P[v]]
        rows.append({"variant": v, "CV macro F1": np.mean(f), "std": np.std(f), "dishes F1": np.mean(fd), "trash F1": np.mean(ft)})
        wrong[v] = {f"#{ids[j]}": int((P[v][:, j] != y[j]).sum()) for j in orig}
    res = pd.DataFrame(rows); res.to_csv(A.ROOT / "logs/bias_cv.csv", index=False)
    w = pd.DataFrame(wrong); w = w[(w > 0).any(axis=1)]; w.to_csv(A.ROOT / "logs/bias_cv_photos.csv")
    b = np.array(B)
    pd.set_option("display.width", 200)
    print("\n" + res.round(3).to_string(index=False))
    print(f"\nchosen bias: dishes mean {b[:, 0].mean():+.2f} (range {b[:, 0].min():+.2f}..{b[:, 0].max():+.2f}), "
          f"trash mean {b[:, 1].mean():+.2f} (range {b[:, 1].min():+.2f}..{b[:, 1].max():+.2f})")
    print("\nPhotos wrong at least once (times wrong out of 10):\n" + w.to_string())


if __name__ == "__main__":
    main()
