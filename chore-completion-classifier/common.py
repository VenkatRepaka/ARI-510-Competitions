"""Shared code for the experiment notebooks (01-07): data, cached features, models and cross-validation.

Task: 4 classes = chore (Washing Dishes / Taking Out Trash) x state (Complete / Incomplete), scored by macro F1.
Data:  data/train_original.csv (66 labelled competition photos), data/test.csv (36 unlabelled competition photos),
       data/external.csv (every external photo ever tried: id, label, group, licence, in_final).
Images: data/images/image_<id>.jpg (competition) and data/images/image_<id>_external.jpg (external).
Features (cached .npz files in features/, made by extract_features.py and notebook 05):
       <backbone>.npz         whole-photo embedding: ids, img, img_flip (mirror), crops (5 crops)
       dinov2_l_tiles.npz     DINOv2 embedding of 13 tiles per photo (2x2 + 3x3 grid)
       overflow_tiles.npz     CDCM overflowing-bin helper, score of each tile (notebook 05)
       bag_learned.npz        learned open-vs-tied bag helper, score of each tile (notebook 05)

Evaluation protocol (all notebooks): 10 x repeated stratified 5-fold CV over the 66 competition photos, grouped by
photo (a photo and anything derived from it are always on the same side of a split). External photos are only ever
added to the training side, and only the 66 competition photos are scored, so every number is comparable.
Reference: scikit-learn RepeatedStratifiedKFold; macro F1 = mean of the per-class F1 scores.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

ROOT = Path(__file__).parent
DATA, FEAT, IMG = ROOT / "data", ROOT / "features", ROOT / "data" / "images"
CLASSES = ["Washing Dishes|Complete", "Washing Dishes|Incomplete", "Taking Out Trash|Complete", "Taking Out Trash|Incomplete"]
BACKBONES = ["dinov2_l", "siglip_so400m", "eva02_l", "vith14_laion", "vitl14_openai"]
REPEATS = 10
torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def competition():
    """(train, test): the 66 labelled and 36 unlabelled competition photos."""
    return pd.read_csv(DATA / "train_original.csv"), pd.read_csv(DATA / "test.csv")


def external():
    return pd.read_csv(DATA / "external.csv")


def training_set(groups=(), final=False, drop=(), relabel=None):
    """66 competition photos + external photos from the named groups of data/external.csv (or the final set).
    relabel = {id: label} changes a label (e.g. {913: 'Washing Dishes|Complete'})."""
    train, _ = competition()
    e = external()
    pick = e[e.in_final] if final else e[e.group.isin(groups)]
    pick = pick[~pick.id.isin(drop)]
    df = pd.concat([train.assign(external=False), pick[["id", "image_file", "label"]].assign(external=True)],
                   ignore_index=True)
    for i, lab in (relabel or {}).items():
        df.loc[df.id == i, "label"] = lab
    df["y"] = df.label.map(CLASSES.index)
    return df


def image_path(i):
    p = IMG / f"image_{i:03d}.jpg"
    return p if p.exists() else IMG / f"image_{i}_external.jpg"


def show(ids, labels=None, h=180, cols=6):
    """Contact sheet of photos (for display in a notebook)."""
    tiles = []
    for k, i in enumerate(ids):
        im = ImageOps.exif_transpose(Image.open(image_path(int(i)))).convert("RGB")
        im.thumbnail((int(h * 1.5), h))
        c = Image.new("RGB", (int(h * 1.5), h + 18), "white")
        c.paste(im, (0, 18))
        ImageDraw.Draw(c).text((3, 3), f"#{i} {labels[k] if labels is not None else ''}", fill="black")
        tiles.append(c)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (int(h * 1.5) + 6), rows * (h + 24)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * (int(h * 1.5) + 6), (k // cols) * (h + 24)))
    return sheet


# ---------------------------------------------------------------- cached features
def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _load(name):
    z = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    return z, {int(i): k for k, i in enumerate(z["ids"])}


def embed(ids, backbone="dinov2_l", view="full"):
    """Whole-photo embedding. view: 'full' = photo + mirror averaged; 'mean7' = average of photo, mirror, 5 crops."""
    z, pos = _load(backbone)
    idx = [pos[int(i)] for i in ids]
    if view == "full":
        return unit(z["img"][idx] + z["img_flip"][idx])
    return unit(np.concatenate([z["img"][idx][:, None], z["img_flip"][idx][:, None], z["crops"][idx]], 1).mean(1))


# 13 overlapping tiles per photo: a 2x2 grid and a 3x3 grid, as (left, top, right, bottom) fractions
TILE_BOXES = [(c / g, r / g, (c + 1) / g, (r + 1) / g) for g in (2, 3) for r in range(g) for c in range(g)]


def tiles_of(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((1536, 1536))
    w, h = im.size
    return [im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))) for x0, y0, x1, y1 in TILE_BOXES]


@torch.no_grad()
def tile_embeddings(ids, backbone="dinov2_l"):
    """Embedding of the 13 tiles of each photo, cached in features/<backbone>_tiles.npz; only new photos are encoded."""
    path = FEAT / f"{backbone}_tiles.npz"
    have = {}
    if path.exists():
        z = np.load(path)
        have = {int(i): t for i, t in zip(z["ids"], z["tiles"])}
    todo = [int(i) for i in ids if int(i) not in have]
    if todo:
        from extract_features import load_model
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        enc, pre = load_model(backbone, device)
        for i in todo:
            have[i] = unit(enc(torch.stack([pre(t) for t in tiles_of(image_path(i))]).to(device))).astype(np.float32)
        keep = sorted(have)
        np.savez(path, ids=np.array(keep), tiles=np.stack([have[i] for i in keep]))
    return np.stack([have[int(i)] for i in ids])


def overflow_inputs(ids):
    """4 numbers from the CDCM overflow helper: most overflowing tile and mean of the top 2, for 2 image models."""
    z, pos = _load("overflow_tiles")
    keys = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    return np.stack([z[k][[pos[int(i)] for i in ids]] for k in keys], 1)


def bag_inputs(ids):
    """2 numbers from the learned open-vs-tied bag helper: most open-looking tile and mean of the top 2."""
    z, pos = _load("bag_learned")
    return np.stack([z[k][[pos[int(i)] for i in ids]] for k in ("bag_open_max", "bag_open_top2")], 1)


def final_features(ids):
    """The final model's input: DINOv2 (2048) + overflow helper (4) + bag helper (2) = 2054 numbers per photo."""
    return np.hstack([embed(ids), overflow_inputs(ids), bag_inputs(ids)])


# ---------------------------------------------------------------- models
class TorchMLP:
    """Small neural net on the embeddings: dropout 0.2 -> Linear(D, 256) -> GELU -> dropout 0.5 -> Linear(256, 4),
    class-weighted cross-entropy with label smoothing 0.1, AdamW, cosine schedule, 600 steps of batch 32, seed 0."""

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
    """Class probabilities; for the SVM a softmax over its one-vs-rest decision values."""
    return clf.predict_proba(X) if hasattr(clf, "predict_proba") else softmax(clf.decision_function(X), axis=1)


# ---------------------------------------------------------------- cross-validation
def splits(df, repeats=REPEATS):
    """(train rows, scored rows) for 10 x 5 folds: competition photos are split, external rows always train."""
    orig, ext = np.where(~df.external.values)[0], np.where(df.external.values)[0]
    y = df.y.values
    return [(np.r_[orig[tr], ext], orig[va]) for tr, va in
            RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0).split(orig, y[orig])]


def cross_validate(df, X, models=("RBF SVM",), extra=None, name=""):
    """CV of each model on feature matrix X (one row per df row).
    extra(train_rows) -> (X_extra, y_extra) adds rows (e.g. augmented views of the training photos) to each fit.
    Returns (summary DataFrame, {model: P}) with P[r, i] = predicted class of row i in repeat r."""
    y = df.y.values
    orig = np.where(~df.external.values)[0]
    dsh, trs = orig[y[orig] < 2], orig[y[orig] >= 2]
    rows, preds = [], {}
    for m in models:
        P = np.full((REPEATS, len(df)), -1)
        for k, (tr, va) in enumerate(splits(df)):
            Xtr, ytr = X[tr], y[tr]
            if extra is not None:
                Xe, ye = extra(tr)
                Xtr, ytr = np.vstack([Xtr, Xe]), np.concatenate([ytr, ye])
            P[k // 5, va] = proba(MODELS[m]().fit(Xtr, ytr), X[va]).argmax(1)
        f = [f1_score(y[orig], p[orig], average="macro") for p in P]
        fd = [f1_score(y[dsh], p[dsh], labels=[0, 1], average="macro") for p in P]
        ft = [f1_score(y[trs], p[trs], labels=[2, 3], average="macro") for p in P]
        rows.append({"experiment": name, "model": m, "CV macro F1": round(np.mean(f), 3), "std": round(np.std(f), 3),
                     "dishes F1": round(np.mean(fd), 3), "trash F1": round(np.mean(ft), 3)})
        preds[m] = P
    return pd.DataFrame(rows), preds


def wrong_counts(df, P):
    """Competition photos predicted wrong at least once: {'#id': times wrong out of 10}."""
    y = df.y.values
    out = {f"#{df.id.iloc[j]}": int((P[:, j] != y[j]).sum()) for j in np.where(~df.external.values)[0]}
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------- duplicates of competition photos
def duplicate_check(ids, threshold=0.90):
    """Nearest competition photo (train or test) for each id by DINOv2 cosine; >= threshold means the same photo.
    Stock-photo sites (Unsplash, Pexels) contain many of the competition photos, so every new photo is checked."""
    train, test = competition()
    comp = list(train.id) + list(test.id)
    C, E = embed(comp), embed(ids)
    S = E @ C.T
    rows = []
    for k, i in enumerate(ids):
        j = int(S[k].argmax())
        rows.append({"id": int(i), "nearest": f"#{comp[j]} ({'test' if comp[j] in set(test.id) else 'train'})",
                     "cosine": round(float(S[k, j]), 3), "duplicate": bool(S[k, j] >= threshold)})
    return pd.DataFrame(rows)
