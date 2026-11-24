"""Does the auxiliary "dirty bin" score (overflow_aux.py) help the chore classifier?

Same protocol as the comparison notebooks, but 10 repeats of photo-grouped stratified 5-fold CV:
the 66 competition photos are scored, external photos in data/train.csv only ever sit in training folds.
Each base feature set is run with and without the 4 overflow scores appended (scaled by ALPHA).
Also reports how well the overflow score alone separates Trash Complete from Trash Incomplete (no training on it).

Writes logs/aux_cv.csv and logs/aux_cv_photos.csv.  Usage: python aux_cv.py
With --tiles: adds the 'dirtiest tile' scores from overflow_aux.py tiles -> logs/aux_cv_tiles.csv, logs/aux_cv_tiles_photos.csv.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

from extract_features import CLASSES, unit

ROOT = Path(__file__).parent
REPEATS, ALPHA = 10, 1.0
BACKBONES = ["dinov2_l", "eva02_l", "siglip_so400m", "vith14_laion", "vitl14_openai"]
WATCH = [19, 63, 72, 80, 100, 26, 41, 77, 12, 2]
torch.set_num_threads(4)

# TRAIN_CSV (default data/train.csv) lets add_batch.py test a candidate training set without editing train.csv
TRAIN_CSV = os.environ.get("TRAIN_CSV", "data/train.csv")
TAG = "" if TRAIN_CSV == "data/train.csv" else "_" + Path(TRAIN_CSV).stem
train = pd.read_csv(ROOT / TRAIN_CSV)
y = train.label.map(CLASSES.index).values
is_ext = train.image_file.str.contains("_external").values
orig, ext = np.where(~is_ext)[0], np.where(is_ext)[0]


def full(b):
    z = np.load(ROOT / f"features/{b}.npz")
    pos = {i: k for k, i in enumerate(z["ids"])}
    idx = [pos[i] for i in train.id]
    return unit(z["img"][idx] + z["img_flip"][idx])


s = np.load(ROOT / "features/overflow_score.npz")
spos = {i: k for k, i in enumerate(s["ids"])}
AUX_KEYS = [k for k in s.files if k != "ids"]
AUX = np.stack([s[k][[spos[i] for i in train.id]] for k in AUX_KEYS], 1)       # (N, 4) in [0, 1]

BASE = {"dinov2 full": full("dinov2_l"), "siglip full": full("siglip_so400m"),
        "5-backbone concat": unit(np.hstack([full(b) for b in BACKBONES]))}


class TorchMLP:
    """Same network as in the notebooks: dropout 0.2 -> Linear(D,256) -> GELU -> dropout 0.5 -> Linear(256,4)."""

    def __init__(self, steps=600, batch=32, seed=0):
        self.steps, self.batch, self.seed = steps, batch, seed

    def fit(self, X, yy):
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 4))
        w = torch.tensor(len(yy) / (4 * np.bincount(yy, minlength=4).clip(1)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        Xt, yt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32), torch.tensor(yy)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.steps):
            b = torch.randint(len(Xt), (min(self.batch, len(Xt)),), generator=g)
            opt.zero_grad(); loss_fn(self.net(Xt[b]), yt[b]).backward(); opt.step(); sched.step()
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)), 1).numpy()


BAL = dict(class_weight="balanced")
MODELS = {"logistic regression": lambda: LogisticRegression(C=1, max_iter=5000, **BAL),
          "RBF SVM": lambda: SVC(kernel="rbf", C=3, gamma="scale", **BAL),
          "extra trees": lambda: ExtraTreesClassifier(300, max_features="sqrt", n_jobs=-1, random_state=0, **BAL),
          "neural net (MLP)": lambda: TorchMLP()}


def proba(clf, X):
    return clf.predict_proba(X) if hasattr(clf, "predict_proba") else softmax(clf.decision_function(X), axis=1)


def main():
    # 1. the overflow score on its own: does it separate Trash Complete (2) from Trash Incomplete (3)?
    trash = orig[np.isin(y[orig], [2, 3])]
    print("Overflow score alone, Trash Complete vs Incomplete on the 33 original trash photos (AUC, 0.5 = chance):")
    for k, key in enumerate(AUX_KEYS):
        print(f"  {key:22s} AUC {roc_auc_score(y[trash] == 3, AUX[trash, k]):.3f}")

    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    rows, photos = [], {}
    for bname, X0 in BASE.items():
        for with_aux in (False, True):
            X = np.hstack([X0, ALPHA * AUX]) if with_aux else X0
            for mname, make in MODELS.items():
                P = np.zeros((REPEATS, len(train), 4))
                for k, (tr, va) in enumerate(splits):
                    P[k // 5, va] = proba(make().fit(X[tr], y[tr]), X[va])
                f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
                key = f"{bname} | {mname} | {'+ overflow score' if with_aux else 'base'}"
                rows.append({"features": bname, "model": mname, "overflow score": with_aux,
                             "CV macro F1": np.mean(f1), "std": np.std(f1)})
                photos[key] = {f"#{w}": f"{P[:, j, y[j]].mean():.2f} ({(P[:, j].argmax(1) != y[j]).sum()}/{REPEATS})"
                               for w in WATCH for j in [int(np.where(train.id == w)[0][0])]}
                print(f"{key:70s} {np.mean(f1):.3f} ± {np.std(f1):.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "logs/aux_cv.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT / "logs/aux_cv_photos.csv")
    t = res.pivot_table(index=["features", "model"], columns="overflow score", values="CV macro F1")
    t.columns = ["base", "+ overflow score"]
    t["change"] = t["+ overflow score"] - t["base"]
    pd.set_option("display.width", 200)
    print("\n" + t.round(3).to_string())
    print(f"\nmean change {t.change.mean():+.3f}; improved {(t.change > 0).sum()} of {len(t)}")


def main_tiles():
    """Zoom: add the 'dirtiest tile' scores (overflow_aux.py tiles) and compare with the whole-image overflow score.
    Same splits and seeds as main(), so results line up with logs/aux_cv.csv."""
    t = np.load(ROOT / "features/overflow_tiles.npz")
    tpos = {i: k for k, i in enumerate(t["ids"])}
    keys = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    TILE = np.stack([t[k][[tpos[i] for i in train.id]] for k in keys], 1)
    trash = orig[np.isin(y[orig], [2, 3])]
    print("Tile scores alone, Trash Complete vs Incomplete on the 33 original trash photos (AUC):")
    for k, key in enumerate(keys):
        print(f"  {key:26s} AUC {roc_auc_score(y[trash] == 3, TILE[trash, k]):.3f}")
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    variants = {"+ tiles": lambda X0: np.hstack([X0, ALPHA * TILE]),
                "+ overflow score + tiles": lambda X0: np.hstack([X0, ALPHA * AUX, ALPHA * TILE])}
    rows, photos = [], {}
    for bname in ("dinov2 full", "siglip full"):
        for vname, build in variants.items():
            X = build(BASE[bname])
            for mname, make in MODELS.items():
                P = np.zeros((REPEATS, len(train), 4))
                for k, (tr, va) in enumerate(splits):
                    P[k // 5, va] = proba(make().fit(X[tr], y[tr]), X[va])
                f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
                key = f"{bname} | {mname} | {vname}"
                rows.append({"features": bname, "model": mname, "variant": vname, "CV macro F1": np.mean(f1), "std": np.std(f1)})
                photos[key] = {f"#{w}": f"{P[:, j, y[j]].mean():.2f} ({(P[:, j].argmax(1) != y[j]).sum()}/{REPEATS})"
                               for w in WATCH + [13, 101] for j in [int(np.where(train.id == w)[0][0])]}
                print(f"{key:70s} {np.mean(f1):.3f} ± {np.std(f1):.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "logs/aux_cv_tiles.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT / "logs/aux_cv_tiles_photos.csv")
    prev = pd.read_csv(ROOT / "logs/aux_cv.csv")
    prev["variant"] = np.where(prev["overflow score"], "+ overflow score", "base")
    allr = pd.concat([prev[prev.features.isin(["dinov2 full", "siglip full"])][res.columns], res])
    tab = allr.pivot_table(index=["features", "model"], columns="variant", values="CV macro F1")
    pd.set_option("display.width", 200)
    print("\n" + tab[["base", "+ overflow score", "+ tiles", "+ overflow score + tiles"]].round(3).to_string())


def main_bags():
    """'Packed bag' hint (bag_tiles.py): add the best bag-like tile score to DINOv2 + overflow tiles (the submitted model)."""
    t = np.load(ROOT / "features/overflow_tiles.npz")
    tpos = {i: k for k, i in enumerate(t["ids"])}
    TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    TILE = np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1)
    b = np.load(ROOT / "features/bag_tiles.npz")
    bpos = {i: k for k, i in enumerate(b["ids"])}
    BAG = np.stack([b[k][[bpos[i] for i in train.id]] for k in ("bag_tile_max", "bag_tile_top2")], 1)
    trash = orig[np.isin(y[orig], [2, 3])]
    print("Packed-bag score alone, Trash Complete vs Incomplete on the 33 original trash photos "
          "(AUC; > 0.5 means higher score = more often Complete):")
    for k, key in enumerate(("bag_tile_max", "bag_tile_top2")):
        print(f"  {key:16s} AUC {roc_auc_score(y[trash] == 2, BAG[trash, k]):.3f}")
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    watch = [26, 72, 19, 80, 84, 100, 63, 13, 77, 41, 2]
    rows, photos = [], {}
    for vname, X in (("dinov2 + tiles", np.hstack([BASE["dinov2 full"], ALPHA * TILE])),
                     ("dinov2 + tiles + bag", np.hstack([BASE["dinov2 full"], ALPHA * TILE, ALPHA * BAG]))):
        for mname, make in MODELS.items():
            P = np.zeros((REPEATS, len(train), 4))
            for k, (tr, va) in enumerate(splits):
                P[k // 5, va] = proba(make().fit(X[tr], y[tr]), X[va])
            f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
            rows.append({"variant": vname, "model": mname, "CV macro F1": np.mean(f1), "std": np.std(f1)})
            photos[f"{vname} | {mname}"] = {f"#{w}": f"{P[:, j, y[j]].mean():.2f} ({(P[:, j].argmax(1) != y[j]).sum()}/{REPEATS})"
                                            for w in watch for j in [int(np.where(train.id == w)[0][0])]}
            print(f"{vname:22s} | {mname:20s} {np.mean(f1):.3f} ± {np.std(f1):.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "logs/aux_cv_bags.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT / "logs/aux_cv_bags_photos.csv")
    pd.set_option("display.width", 200)
    print("\n" + res.pivot_table(index="model", columns="variant", values="CV macro F1").round(3).to_string())
    print("\n" + pd.DataFrame(photos)[[c for c in photos if "RBF SVM" in c]].to_string())


def main_open():
    """Open-bag hint (bag_tiles.py open_tile_*): add it to DINOv2 + overflow tiles (the submitted model).
    The 'dinov2 + tiles' row re-runs the submitted model on the current train.csv, so the open-bag photos'
    effect as plain training data and as a feature can be read separately."""
    t = np.load(ROOT / "features/overflow_tiles.npz")
    tpos = {i: k for k, i in enumerate(t["ids"])}
    TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    TILE = np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1)
    b = np.load(ROOT / "features/bag_tiles.npz")
    bpos = {i: k for k, i in enumerate(b["ids"])}
    OPEN = np.stack([b[k][[bpos[i] for i in train.id]] for k in ("open_tile_max", "open_tile_top2")], 1)
    OPEN = 4.0 * OPEN          # raw spread (std 0.05) x4 ~ the spread of the overflow tile scores (std 0.15-0.24)
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    watch = [72, 41, 77, 26, 19, 80, 100, 63, 13, 2, 12, 101]
    rows, photos = [], {}
    for vname, X in (("dinov2 + tiles", np.hstack([BASE["dinov2 full"], ALPHA * TILE])),
                     ("dinov2 + tiles + open", np.hstack([BASE["dinov2 full"], ALPHA * TILE, ALPHA * OPEN]))):
        for mname, make in MODELS.items():
            P = np.zeros((REPEATS, len(train), 4))
            for k, (tr, va) in enumerate(splits):
                P[k // 5, va] = proba(make().fit(X[tr], y[tr]), X[va])
            f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
            # per chore: macro F1 over that chore's two classes, on that chore's competition photos
            dsh, trs = orig[y[orig] < 2], orig[y[orig] >= 2]
            f1_d = [f1_score(y[dsh], p[dsh].argmax(1), labels=[0, 1], average="macro") for p in P]
            f1_t = [f1_score(y[trs], p[trs].argmax(1), labels=[2, 3], average="macro") for p in P]
            rows.append({"variant": vname, "model": mname, "CV macro F1": np.mean(f1), "std": np.std(f1),
                         "dishes F1": np.mean(f1_d), "trash F1": np.mean(f1_t)})
            photos[f"{vname} | {mname}"] = {f"#{w}": f"{P[:, j, y[j]].mean():.2f} ({(P[:, j].argmax(1) != y[j]).sum()}/{REPEATS})"
                                            for w in watch for j in [int(np.where(train.id == w)[0][0])]}
            print(f"{vname:24s} | {mname:20s} {np.mean(f1):.3f} ± {np.std(f1):.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / f"logs/aux_cv_open{TAG}.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT / f"logs/aux_cv_open{TAG}_photos.csv")
    pd.set_option("display.width", 200)
    print("\n" + res.pivot_table(index="model", columns="variant", values="CV macro F1").round(3).to_string())
    print("\n" + pd.DataFrame(photos)[[c for c in photos if "RBF SVM" in c or "logistic" in c]].to_string())


if __name__ == "__main__":
    import sys
    (main_open() if "--open" in sys.argv else main_bags() if "--bags" in sys.argv
     else main_tiles() if "--tiles" in sys.argv else main())
