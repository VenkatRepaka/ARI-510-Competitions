"""Dirty-dish tile feature built only from our own training photos (no external data).

1. encode : every hand-drawn box in dish_boxes.py (16 dirty from Incomplete, 17 clean from Complete photos) is
            augmented N_AUG times (random crop of the box with 0.7-1.4x context, small rotation, flip,
            brightness/contrast/saturation jitter, sometimes grey or blur) and embedded with DINOv2
            -> features/dish_crops.npz (ids, y, feats (33, N_AUG, D))
2. tiles4 : a 4x4 grid of DINOv2 tile embeddings for every photo (#2's plate is smaller than a 3x3 tile)
            -> features/dinov2_l_tiles4.npz; used together with the 13 saved tiles (29 tiles per photo)
3. cv     : SAME splits/model as aux_cv / dishes_errors (DINOv2 full + overflow tiles + open-bag, RBF SVM).
            Inside each fold, a logistic regression is trained on the crops of that fold's TRAINING photos only,
            scores all 29 tiles of every photo, and dirty_max / dirty_top2 are appended as features (x SCALE).
            Prints overall + dishes F1 and per-photo error counts -> logs/dish_crops_cv.csv
4. final  : classifier on all 33 crops -> features/dish_tiles.npz (for a later submission)
Usage: python dish_crops.py [encode] [tiles4] [cv] [final]   (default: all)
"""
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

import aux_cv as A
from dish_boxes import CLEAN, DIRTY
from extract_features import OUT, get_logger, image_paths, load_model, unit

N_AUG, SEED = 24, 0
BOXES4 = [(c / 4, r / 4, (c + 1) / 4, (r + 1) / 4) for r in range(4) for c in range(4)]
log = get_logger("dish_crops", "dish_crops.log")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def open_img(i, paths):
    im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
    im.thumbnail((1536, 1536))
    return im


def augment(im, box, rng):
    w, h = im.size
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    s = rng.uniform(0.7, 1.4)
    cx += rng.uniform(-0.1, 0.1) * bw; cy += rng.uniform(-0.1, 0.1) * bh
    c = im.crop((int(max(0, cx - s * bw / 2)), int(max(0, cy - s * bh / 2)),
                 int(min(w, cx + s * bw / 2)), int(min(h, cy + s * bh / 2))))
    c = c.rotate(rng.uniform(-15, 15), resample=Image.BILINEAR, expand=False)
    if rng.random() < .5:
        c = ImageOps.mirror(c)
    from PIL import ImageEnhance as E
    c = E.Brightness(c).enhance(rng.uniform(.6, 1.4))
    c = E.Contrast(c).enhance(rng.uniform(.7, 1.3))
    c = E.Color(c).enhance(rng.uniform(.6, 1.4))
    if rng.random() < .15:
        c = ImageOps.grayscale(c).convert("RGB")
    if rng.random() < .2:
        c = c.filter(ImageFilter.GaussianBlur(rng.uniform(.5, 2)))
    return c


@torch.no_grad()
def encode():
    enc, pre = load_model("dinov2_l", DEVICE)
    paths, rng = image_paths(), np.random.default_rng(SEED)
    ids, y, feats = [], [], []
    for lab, B in ((1, DIRTY), (0, CLEAN)):
        for i, box in B.items():
            im = open_img(i, paths)
            views = [augment(im, box, rng) for _ in range(N_AUG)]
            feats.append(unit(enc(torch.stack([pre(v) for v in views]).to(DEVICE))).astype(np.float32))
            ids.append(i); y.append(lab)
    np.savez(OUT / "dish_crops.npz", ids=np.array(ids), y=np.array(y), feats=np.stack(feats))
    log.info(f"saved features/dish_crops.npz {np.stack(feats).shape}")


@torch.no_grad()
def tiles4():
    ids = sorted(set(pd.read_csv(A.ROOT / "data/train.csv").id) | set(pd.read_csv(A.ROOT / "data/test.csv").id))
    enc, pre = load_model("dinov2_l", DEVICE)
    paths, out = image_paths(), []
    for k, i in enumerate(ids, 1):
        im = open_img(i, paths); w, h = im.size
        ts = [im.crop((int(a * w), int(b * h), int(c * w), int(d * h))) for a, b, c, d in BOXES4]
        out.append(unit(enc(torch.stack([pre(t) for t in ts]).to(DEVICE))).astype(np.float32))
        if k % 25 == 0:
            log.info(f"tiles4 {k}/{len(ids)}")
    np.savez(OUT / "dinov2_l_tiles4.npz", ids=np.array(ids), tiles=np.stack(out))
    log.info(f"saved features/dinov2_l_tiles4.npz {np.stack(out).shape}")


def all_tiles(ids):
    a, b = np.load(OUT / "dinov2_l_tiles.npz"), np.load(OUT / "dinov2_l_tiles4.npz")
    pa, pb = {int(i): k for k, i in enumerate(a["ids"])}, {int(i): k for k, i in enumerate(b["ids"])}
    return np.concatenate([a["tiles"][[pa[i] for i in ids]], b["tiles"][[pb[i] for i in ids]]], 1)   # (N, 29, D)


def crop_clf(keep_ids):
    z = np.load(OUT / "dish_crops.npz")
    m = np.isin(z["ids"], list(keep_ids))
    F = z["feats"][m]
    return LogisticRegression(C=1, max_iter=5000).fit(F.reshape(-1, F.shape[-1]), np.repeat(z["y"][m], F.shape[1]))


def dirty_feats(clf, T):
    s = clf.predict_proba(T.reshape(-1, T.shape[-1]))[:, 1].reshape(len(T), -1)
    return np.stack([s.max(1), np.sort(s, 1)[:, -2:].mean(1)], 1), s


def base_X():
    t = np.load(A.ROOT / "features/overflow_tiles.npz"); tp = {i: k for k, i in enumerate(t["ids"])}
    TILE = np.stack([t[k][[tp[i] for i in A.train.id]] for k in
                     ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]], 1)
    b = np.load(A.ROOT / "features/bag_tiles.npz"); bp = {i: k for k, i in enumerate(b["ids"])}
    OPEN = 4.0 * np.stack([b[k][[bp[i] for i in A.train.id]] for k in ("open_tile_max", "open_tile_top2")], 1)
    return np.hstack([A.BASE["dinov2 full"], TILE, OPEN])


def cv():
    X0, y, orig, ext = base_X(), A.y, A.orig, A.ext
    ids = A.train.id.values
    T = all_tiles(list(ids))
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig]))
    # per split: dirty features from a crop classifier that saw only that split's training photos
    D = []
    for tr, va in splits:
        D.append(dirty_feats(crop_clf(set(ids[orig[tr]])), T)[0])
    dishes = orig[np.isin(y[orig], (0, 1))]
    rows, per_photo = [], {}
    for name, scale in (("base", None), ("+dirty x1", 1.0), ("+dirty x2", 2.0), ("+dirty x4", 4.0)):
        P = np.zeros((A.REPEATS, len(y)), int)
        for k, (tr, va) in enumerate(splits):
            X = X0 if scale is None else np.hstack([X0, scale * D[k]])
            tr = np.r_[orig[tr], ext]; va = orig[va]
            P[k // 5, va] = A.proba(A.MODELS["RBF SVM"]().fit(X[tr], y[tr]), X[va]).argmax(1)
        f = [f1_score(y[orig], P[r, orig], average="macro") for r in range(A.REPEATS)]
        fd = [f1_score(y[dishes], P[r, dishes], average="macro") for r in range(A.REPEATS)]
        wrong = {int(ids[j]): int((P[:, j] != y[j]).sum()) for j in orig if (P[:, j] != y[j]).any()}
        rows.append({"model": name, "f1": round(np.mean(f), 3), "sd": round(np.std(f), 3), "dishes_f1": round(np.mean(fd), 3),
                     "wrong (id: n/10)": wrong})
        per_photo[name] = wrong
        log.info(f"{name:10s} F1 {np.mean(f):.3f} ± {np.std(f):.3f}  dishes {np.mean(fd):.3f}  wrong {wrong}")
    pd.DataFrame(rows).to_csv(A.ROOT / "logs/dish_crops_cv.csv", index=False)
    # in-fold dirty score of each dishes photo (from the fold where it was held out, repeat 1)
    hold = np.zeros(len(y))
    for k, (tr, va) in enumerate(splits[:5]):
        hold[orig[va]] = D[k][orig[va], 0]
    lab = A.train.label.values
    log.info("held-out dirty_max per dishes photo: " + ", ".join(
        f"#{ids[j]} {'I' if lab[j].endswith('Incomplete') else 'C'} {hold[j]:.2f}" for j in sorted(dishes, key=lambda j: hold[j])))


def final():
    ids = sorted(set(pd.read_csv(A.ROOT / "data/train.csv").id) | set(pd.read_csv(A.ROOT / "data/test.csv").id))
    f, s = dirty_feats(crop_clf(set(DIRTY) | set(CLEAN)), all_tiles(ids))
    np.savez(OUT / "dish_tiles.npz", ids=np.array(ids), dirty_max=f[:, 0], dirty_top2=f[:, 1], tile_scores=s)
    log.info(f"saved features/dish_tiles.npz for {len(ids)} photos")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["encode", "tiles4", "cv", "final"]
    for st in ("encode", "tiles4", "cv", "final"):
        if st in steps:
            globals()[st]()
