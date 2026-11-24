"""Sink rule: 'anything in or on the sink = Incomplete'. Complete dishes photos that break it are relabelled
Incomplete in the TRAINING folds only, then held-out photos are scored against (a) the host's labels and (b) the
sink-rule labels. Same 10x5 photo-grouped splits, frozen DINOv2 full + RBF SVM (4 classes).
-> logs/relabel_sink_cv.txt.  Usage: python relabel_sink_cv.py
"""
import numpy as np
from sklearn.metrics import f1_score

import aux_cv as A

SETS = {"host labels (no change)": [], "#101": [101], "#101 #64": [101, 64], "#101 #64 #40 #99": [101, 64, 40, 99]}
ids = A.train.id.values
X, y0 = A.BASE["dinov2 full"], A.y
splits = list(A.RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(A.orig, A.y[A.orig]))
lines = []
for name, rel in SETS.items():
    y1 = y0.copy(); y1[np.isin(ids, rel)] = 1                      # 1 = Washing Dishes|Incomplete
    P = np.zeros((A.REPEATS, len(y0)), int)
    for k, (tr, va) in enumerate(splits):
        trr = np.r_[A.orig[tr], A.ext]
        P[k // 5, A.orig[va]] = A.MODELS["RBF SVM"]().fit(X[trr], y1[trr]).predict(X[A.orig[va]])
    o = A.orig; d = np.isin(y0[o], [0, 1])
    for tag, yt in (("vs host", y0), ("vs sink rule", y1)):
        f = [f1_score(yt[o], P[r, o], average="macro") for r in range(A.REPEATS)]
        fd = [f1_score(yt[o][d], P[r, o][d], average="macro", labels=[0, 1]) for r in range(A.REPEATS)]
        wrong = {int(ids[j]): int(n) for j, n in zip(o, (P[:, o] != yt[o]).sum(0)) if n and y0[j] in (0, 1)}
        lines.append(f"{name:24s} {tag:13s} F1 {np.mean(f):.3f} ± {np.std(f):.3f}  dishes {np.mean(fd):.3f}  dishes wrong {wrong}")
txt = "\n".join(lines); print(txt); (A.ROOT / "logs/relabel_sink_cv.txt").write_text(txt + "\n")
