"""Leave-one-external-out for the current candidate (DINOv2 + tiles + open, RBF SVM, 10 reps): effect on #13 and CV."""
import numpy as np, pandas as pd
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
base = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig]))
watch = [13, 19, 72, 77, 63]
def run(drop):
    keep = np.array([j for j in ext if ids[j] not in drop])
    P = np.zeros((A.REPEATS, len(y), 4))
    for k, (tr, va) in enumerate(base):
        trn = np.r_[orig[tr], keep]
        P[k // 5, orig[va]] = A.proba(A.MODELS["RBF SVM"]().fit(X[trn], y[trn]), X[orig[va]])
    f1 = np.mean([f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P])
    w = {f"#{i}": int((P[:, j].argmax(1) != y[j]).sum()) for i in watch for j in [int(np.where(ids == i)[0][0])]}
    return f1, w
cases_all = {"none": [], "945": [945], "947": [947], "948": [948], "949": [949], "all 4 dumpsters": [945, 947, 948, 949],
         "all 5 open bags": [951, 952, 953, 954, 955]}
cases = cases_all if __import__("sys").argv[1:] == [] else {a: [int(x) for x in a.split("+")] for a in __import__("sys").argv[1:]}
for name, drop in cases.items():
    f1, w = run(drop)
    print(f"drop {name:16s} CV {f1:.3f}  wrong/10: {w}", flush=True)
