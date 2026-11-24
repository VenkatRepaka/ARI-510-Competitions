"""Does a separate dishes-only model help, and do the web sink photos help it?

Run with TRAIN_CSV=train_sinks.csv (66 original photos + 9 web sink photos, ids 914-924).
Compares, with the same 10x repeated 5-fold CV over the 66 original photos (web photos only ever train):
  4-class          one classifier for all 4 labels (current approach)
  per-chore        chore from the 4-class model, then complete/incomplete from a classifier trained
                   only on photos of that chore (dishes-only model for dishes, trash-only for trash)
each with and without the web sink photos, for several classifiers (features: all backbones | full view,
probabilities averaged over backbones). Reports overall macro F1, dishes-only macro F1 and per-photo errors
of the dishes photos the models keep missing.

Usage: TRAIN_CSV=train_sinks.csv python dishes_experiment.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

import train as T

BACKBONES = ["dinov2_l", "eva02_l", "siglip_so400m", "vith14_laion", "vitl14_openai"]
CLFS = ["extra trees", "logreg C=10", "RBF SVM C=3"]
WATCH = [12, 60, 31, 101, 99, 2, 22, 46]   # dishes photos the models keep missing

y, orig, ext = T.y, T.orig_idx, T.ext_idx
chore, inc = y // 2, y % 2
F = {b: T.load(b) for b in BACKBONES}
X = {b: F[b]["tr", "full"][0][:, 0] for b in BACKBONES}   # (n_rows, D)
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(y[orig], y[orig]))


def four_class(clf, tr, va):
    return np.mean([T.CLASSIFIERS[clf](X[b][tr], y[tr], X[b][va], X[b][va][:0]) for b in BACKBONES], 0)


def binary(clf, Xtr, ytr, Xte):
    """P(incomplete). train.py's SVM wrapper expects 4 classes, so the binary SVM is handled here."""
    if clf.startswith("RBF SVM"):
        from scipy.special import expit
        from sklearn.svm import SVC
        return expit(SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced").fit(Xtr, ytr).decision_function(Xte))
    return T.CLASSIFIERS[clf](Xtr, ytr, Xte, Xte[:0])[:, 1]


def per_chore(clf, tr, va, chore_pred):
    p_inc = np.zeros(len(va))
    for c in (0, 1):
        t, v = tr[chore[tr] == c], chore_pred == c
        if v.any():
            p_inc[v] = np.mean([binary(clf, X[b][t], inc[t], X[b][va][v]) for b in BACKBONES], 0)
    return np.stack([(chore_pred == 0) * (1 - p_inc), (chore_pred == 0) * p_inc,
                     (chore_pred == 1) * (1 - p_inc), (chore_pred == 1) * p_inc], 1)


rows, watch_err = [], {}
ids = T.train.id.values
for clf in CLFS:
    for use_web in (False, True):
        for kind in ("4-class", "per-chore"):
            overall, dishes = [], []
            err = {i: 0 for i in WATCH}
            for tr0, va0 in splits:
                tr = np.r_[orig[tr0], ext] if use_web else orig[tr0]
                va = orig[va0]
                p4 = four_class(clf, tr, va)
                if kind == "per-chore":
                    p4 = per_chore(clf, tr, va, (p4[:, 2:].sum(1) > p4[:, :2].sum(1)).astype(int))
                pred = p4.argmax(1)
                overall.append(f1_score(y[va], pred, average="macro"))
                d = chore[va] == 0
                dishes.append(f1_score(y[va][d], pred[d], average="macro"))
                for i, p, t in zip(ids[va], pred, y[va]):
                    if i in err:
                        err[i] += p != t
            name = f"{clf} | {kind} | {'+ web sinks' if use_web else 'no web'}"
            rows.append((name, np.mean(overall), np.std(overall), np.mean(dishes)))
            watch_err[name] = {i: e / 10 for i, e in err.items()}
            print(f"{name:45s} overall {np.mean(overall):.3f} ± {np.std(overall):.3f}   dishes {np.mean(dishes):.3f}",
                  flush=True)

res = pd.DataFrame(rows, columns=["entry", "overall_f1", "std", "dishes_f1"]).sort_values("overall_f1", ascending=False)
res.to_csv(T.LOG_DIR / "dishes_experiment.csv", index=False)
print("\nError rate on the hard dishes photos (share of the 10 CV repeats it was wrong):")
print(pd.DataFrame(watch_err).T.loc[res.entry].to_string())
