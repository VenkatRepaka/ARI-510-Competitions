"""Image-only classifiers on frozen backbone embeddings (no text encoder, no language model).

Stage 1: every (backbone, view mode, classifier) is scored with repeated stratified K-fold CV (macro F1).
Stage 2: ensembles that average the out-of-fold probabilities of several stage-1 runs.
The best entry is refit on all training photos and written to submission.csv.
Every run is logged to the console and to logs/train_<time>.log; a results table goes to logs/.

View modes (what a photo contributes when training / when predicting):
  full     train: mean(image, mirror)               predict: same
  mean7    train: mean of all 7 views               predict: same
  aug      train: image, mirror + N augmented views predict: image, mirror (probabilities averaged)
  aug+tta  train: as aug                            predict: image, mirror + N augmented views
The aug modes need features/<backbone>_aug.npz (python extract_features.py --aug 20 <backbone>).

Usage: python train.py [--repeats 10] [--modes full mean7 aug aug+tta] [--backbones dinov2_l ...]
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC

from extract_features import CLASSES, LOG_DIR, get_logger, unit

ROOT = Path(__file__).parent
train = pd.read_csv(ROOT / "data" / os.environ.get("TRAIN_CSV", "train.csv"))  # TRAIN_CSV picks another train file
test = pd.read_csv(ROOT / "data" / "test.csv")
y = train.label.map(CLASSES.index).values
# Extra photos from the web (image_9NN_external.jpg): always in the training folds, never scored in CV,
# so CV stays comparable with runs on the original photos only.
is_ext = train.image_file.str.contains("_external").values
orig_idx, ext_idx = np.where(~is_ext)[0], np.where(is_ext)[0]
ALL_BACKBONES = [p.stem for p in sorted((ROOT / "features").glob("*.npz"))   # skip _aug, patch and OOF files
                 if not p.stem.endswith("_aug") and "img" in np.load(p, allow_pickle=True).files]
ALL_MODES = ["full", "mean7", "aug", "aug+tta"]
SHORT = ["Dishes ok", "Dishes x", "Trash ok", "Trash x"]


def load(name):
    """Returns {(split, mode): (train_views, predict_views)}, each of shape (n_photos, n_views, D)."""
    z = np.load(ROOT / "features" / f"{name}.npz")
    aug_path = ROOT / "features" / f"{name}_aug.npz"
    za = np.load(aug_path) if aug_path.exists() else None
    pos = {i: k for k, i in enumerate(z["ids"])}
    out = {}
    for split, df in (("tr", train), ("te", test)):
        idx = [pos[i] for i in df.id]
        full, flip, crops = z["img"][idx], z["img_flip"][idx], z["crops"][idx]
        both = unit(full + flip)[:, None]
        mean7 = unit(full + flip + crops.sum(1))[:, None]
        out[split, "full"] = (both, both)
        out[split, "mean7"] = (mean7, mean7)
        if za is not None and set(df.id) <= set(za["ids"]):
            pos_a = {i: k for k, i in enumerate(za["ids"])}
            aug = za["aug"][[pos_a[i] for i in df.id]]
            pair = np.stack([full, flip], 1)
            with_aug = np.concatenate([pair, aug], 1)
            out[split, "aug"] = (with_aug, pair)
            out[split, "aug+tta"] = (with_aug, with_aug)
    return out


class NearestCentroid:
    """Cosine similarity to each class mean, turned into probabilities."""

    def __init__(self, temp=20):
        self.temp = temp

    def fit(self, X, y):
        self.c = unit(np.stack([X[y == k].mean(0) for k in range(4)]))
        return self

    def predict_proba(self, X):
        return softmax(self.temp * X @ self.c.T, axis=1)


def sk(make):
    """Wrap a scikit-learn classifier as fit_predict(Xtr, ytr, Xte, Xunlabeled) -> (n, 4) probabilities."""
    def run(Xtr, ytr, Xte, Xu):
        clf = make().fit(Xtr, ytr)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(Xte)
        return softmax(clf.decision_function(Xte), axis=1)
    return run


def label_spreading(Xtr, ytr, Xte, Xu):
    """Transductive: also uses the unlabeled photos (validation fold + test set) to shape the graph."""
    X = np.vstack([Xtr, Xte, Xu])
    labels = np.r_[ytr, -np.ones(len(Xte) + len(Xu), int)]
    m = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=1000).fit(X, labels)
    return m.label_distributions_[len(Xtr):len(Xtr) + len(Xte)]


BAL = dict(class_weight="balanced")
CLASSIFIERS = {
    "logreg C=1": sk(lambda: LogisticRegression(C=1, max_iter=5000, **BAL)),
    "logreg C=10": sk(lambda: LogisticRegression(C=10, max_iter=5000, **BAL)),
    "linear SVM C=1": sk(lambda: SVC(kernel="linear", C=1, **BAL)),
    "linear SVM C=10": sk(lambda: SVC(kernel="linear", C=10, **BAL)),
    "RBF SVM C=3": sk(lambda: SVC(kernel="rbf", C=3, gamma="scale", **BAL)),
    "RBF SVM C=10": sk(lambda: SVC(kernel="rbf", C=10, gamma="scale", **BAL)),
    "ridge a=1": sk(lambda: RidgeClassifier(alpha=1, **BAL)),
    "PCA32 + LDA": sk(lambda: make_pipeline(PCA(32, random_state=0),
                                            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
    "nearest centroid": sk(NearestCentroid),
    "kNN k=5": sk(lambda: KNeighborsClassifier(5, metric="cosine", weights="distance")),
    "extra trees": sk(lambda: ExtraTreesClassifier(300, max_features="sqrt", n_jobs=-1, random_state=0, **BAL)),
    "MLP 256": sk(lambda: MLPClassifier((256,), alpha=1e-2, max_iter=2000, random_state=0)),
    "label spreading": label_spreading,
}
# ~20x more training rows in the aug modes: these become too slow (MLP) or are a poor fit (graph methods).
SKIP_ON_AUG = {"MLP 256", "label spreading", "kNN k=5"}


def fit_predict(F, backbone, mode, clf_name, tr_idx, pred_split, pred_idx):
    Vtr, _ = F[backbone]["tr", mode]
    _, Vte = F[backbone][pred_split, mode]
    Vte = Vte[pred_idx]
    Xtr = Vtr[tr_idx].reshape(-1, Vtr.shape[-1])
    ytr = np.repeat(y[tr_idx], Vtr.shape[1])
    Xu = F[backbone]["te", mode][1][:, 0] if pred_split == "tr" else np.empty((0, Vtr.shape[-1]))
    p = CLASSIFIERS[clf_name](Xtr, ytr, Vte.reshape(-1, Vte.shape[-1]), Xu)
    return p.reshape(len(Vte), Vte.shape[1], 4).mean(1)


def main(repeats, modes, backbones, entry=None, out="submission.csv"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = get_logger("train", f"train_{stamp}.log")
    t_start = time.time()
    F = {}
    for b in backbones:
        F[b] = load(b)
        have = [m for m in modes if ("tr", m) in F[b]]
        Vtr = F[b]["tr", have[-1]][0]
        log.info(f"[load] {b}: dim {Vtr.shape[-1]}, modes {have}")
    log.info(f"classifiers: {list(CLASSIFIERS)}")
    yo = y[orig_idx]
    splits = [(np.r_[orig_idx[tr], ext_idx], orig_idx[va])
              for tr, va in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0).split(yo, yo)]
    log.info(f"{len(orig_idx)} original photos (scored in CV) + {len(ext_idx)} external photos (train folds only)")

    runs = [(b, m, c) for b in backbones for m in modes for c in CLASSIFIERS
            if ("tr", m) in F[b] and not (m.startswith("aug") and c in SKIP_ON_AUG)]
    log.info(f"===== Stage 1: {len(runs)} runs x {len(splits)} folds =====")
    oof, rows = {}, []
    for k, (b, m, c) in enumerate(runs, 1):
        t0 = time.time()
        try:
            probs = [fit_predict(F, b, m, c, tr, "tr", va) for tr, va in splits]
        except Exception:
            log.exception(f"[{k}/{len(runs)}] {b} | {m} | {c} FAILED")
            continue
        s = [f1_score(y[va], p.argmax(1), average="macro") for (_, va), p in zip(splits, probs)]
        oof[b, m, c] = probs
        rows.append({"entry": f"{b} | {m} | {c}", "kind": "single", "backbone": b, "mode": m, "classifier": c,
                     "cv_macro_f1": np.mean(s), "std": np.std(s), "members": [(b, m, c)]})
        el = time.time() - t_start
        log.info(f"[{k:3d}/{len(runs)}] {b:14s} | {m:7s} | {c:17s} F1 {np.mean(s):.3f} ± {np.std(s):.3f}  "
                 f"({time.time() - t0:.1f}s, eta {el / k * (len(runs) - k) / 60:.1f} min)")

    def score(members):
        s = [f1_score(y[va], np.mean([oof[mb][i] for mb in members], 0).argmax(1), average="macro")
             for i, (_, va) in enumerate(splits)]
        return np.mean(s), np.std(s)

    def add_ensemble(name, members):
        members = [mb for mb in members if mb in oof]
        if len(members) < 2:
            return
        mean, std = score(members)
        rows.append({"entry": name, "kind": "ensemble", "cv_macro_f1": mean, "std": std, "members": members})
        log.info(f"[ensemble] {name:55s} F1 {mean:.3f} ± {std:.3f}  ({len(members)} members)")

    log.info("===== Stage 2: ensembles (average of out-of-fold probabilities) =====")
    for c in CLASSIFIERS:
        for m in modes:
            add_ensemble(f"all backbones | {m} | {c}", [(b, m, c) for b in backbones])
    singles = pd.DataFrame([r for r in rows if r["kind"] == "single"])
    best_per_backbone = singles.loc[singles.groupby("backbone").cv_macro_f1.idxmax()]
    add_ensemble("best run per backbone", [r.members[0] for r in best_per_backbone.itertuples()])
    for n in (3, 5, 10):
        top = singles.nlargest(n, "cv_macro_f1")
        add_ensemble(f"top-{n} single runs (selection-biased)", [r.members[0] for r in top.itertuples()])

    res = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
    res.drop(columns="members").to_csv(LOG_DIR / f"results_{stamp}.csv", index=False)
    log.info("===== Top 20 =====")
    for r in res.head(20).itertuples():
        log.info(f"{r.cv_macro_f1:.3f} ± {r.std:.3f}  {r.entry}")
    log.info("Best per mode (single run):")
    for _, r in singles.loc[singles.groupby("mode").cv_macro_f1.idxmax()].iterrows():
        log.info(f"  {r['mode']:7s} {r.cv_macro_f1:.3f}  {r.backbone} | {r.classifier}")

    best = res[res.entry == entry].iloc[0] if entry else res[~res.entry.str.contains("selection-biased")].iloc[0]
    oof_best = [np.mean([oof[mb][i] for mb in best.members], 0) for i in range(len(splits))]
    cm = sum(confusion_matrix(y[va], p.argmax(1), labels=range(4)) for (_, va), p in zip(splits, oof_best))
    log.info(f"confusion matrix of best, rows = true class (one pass over the {len(orig_idx)} original photos, avg of {repeats}):\n"
             + pd.DataFrame(np.round(cm / repeats).astype(int), SHORT, SHORT).to_string())

    log.info(f"===== Final fit: {best.entry} (CV {best.cv_macro_f1:.3f}) on all {len(y)} photos =====")
    all_tr, all_te = np.arange(len(y)), np.arange(len(test))
    proba = np.mean([fit_predict(F, *mb, all_tr, "te", all_te) for mb in best.members], 0)
    sub = pd.DataFrame({"id": test.id, "label": [CLASSES[k] for k in proba.argmax(1)]})
    sub.to_csv(ROOT / out, index=False)
    log.info(f"wrote {out}: " + ", ".join(f"{k}={v}" for k, v in sub.label.value_counts().items()))
    log.info(f"done in {(time.time() - t_start) / 60:.1f} min. Results table: logs/results_{stamp}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    ap.add_argument("--backbones", nargs="+", default=ALL_BACKBONES, choices=ALL_BACKBONES)
    ap.add_argument("--entry", help="write the submission from this entry instead of the best one, "
                                     "e.g. 'all backbones | full | extra trees'")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    main(a.repeats, a.modes, a.backbones, a.entry, a.out)
