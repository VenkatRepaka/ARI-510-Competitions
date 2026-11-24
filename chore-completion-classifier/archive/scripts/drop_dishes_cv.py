"""Does training without suspect dishes photos help? Same model/splits as dishes_errors.py; dropped photos removed
from training folds only, all 66 originals still scored. Prints overall and dishes macro F1 (10 repeats)."""
import sys, numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
import aux_cv as A
t = np.load(A.ROOT / "features/overflow_tiles.npz"); tp = {i: k for k, i in enumerate(t["ids"])}
TILE = np.stack([t[k][[tp[i] for i in A.train.id]] for k in
                 ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]], 1)
b = np.load(A.ROOT / "features/bag_tiles.npz"); bp = {i: k for k, i in enumerate(b["ids"])}
OPEN = 4.0 * np.stack([b[k][[bp[i] for i in A.train.id]] for k in ("open_tile_max", "open_tile_top2")], 1)
X = np.hstack([A.BASE["dinov2 full"], TILE, OPEN]); y, orig, ext = A.y, A.orig, A.ext
ids = A.train.id.values
SPLITS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig]))
for drop in [[], [60], [31], [101], [60, 31], [60, 31, 101], [60, 31, 101, 2, 99]]:
    keep = ~np.isin(ids, drop)
    f_all, f_d, wrong = [], [], {}
    for r in range(A.REPEATS):
        pred = np.zeros(len(y), int)
        for tr, va in SPLITS[r*5:(r+1)*5]:
            tr = np.r_[orig[tr], ext]; tr = tr[keep[tr]]; va = orig[va]
            pred[va] = A.proba(A.MODELS["RBF SVM"]().fit(X[tr], y[tr]), X[va]).argmax(1)
        f_all.append(f1_score(y[orig], pred[orig], average="macro"))
        d = orig[np.isin(y[orig], (0, 1))]
        f_d.append(f1_score(y[d], pred[d], average="macro"))
        for j in d:
            if pred[j] != y[j]: wrong[ids[j]] = wrong.get(ids[j], 0) + 1
    print(f"drop {str(drop):24s} all {np.mean(f_all):.3f}±{np.std(f_all):.3f}  dishes {np.mean(f_d):.3f}  wrong {dict(sorted(wrong.items()))}")
