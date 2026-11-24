"""Find soiled dishes anywhere in a photo with a sliding-window classifier trained on our own crops.

Windows: two scales of overlapping square-ish windows (side 1/3 of the image, stride 1/6 -> 5x5 = 25;
side 1/4, stride 1/8 -> 7x7 = 49), 74 per photo, DINOv2 embedded (same preprocessing as extract_features.py)
for every dishes photo: 33 original dishes photos, external #912 and the test photos the 4-class SVM calls dishes.
-> features/soiled_windows.npz (ids, boxes (74, 4), emb (N, 74, D))

Window labels, from the soiled boxes in dish_boxes.DIRTY (Incomplete photos; #28 and #31 left out: no visible soil):
  positive      : >= 60% of the window lies in a soiled box, or >= 60% of the box lies in the window
  hard negative : < 10% overlap with every soiled box (background of Incomplete photos), and every window of a
                  Complete photo (clean dishes, racks, sinks, counters)
  ignored       : everything in between, and all windows of #28 / #31
Plus the hand crops themselves (features/dish_crops.npz, 24 augmented views each) as extra positives / negatives.

CV (same dishes splits as dishes_cv.py): inside each fold the window classifier (LR) is trained on the training
photos only, then every held-out photo gets soiled_max (max window score) and soiled_top3 (mean of the top 3).
Reported: AUC of soiled_max alone, dishes LR (+10 aug views) with and without the two scores, routed overall F1.
-> logs/soiled_windows_cv.txt, logs/soiled_windows_maps.jpg (where it fires on the hard photos)
Usage: python soiled_windows.py [encode] [cv]   (default both)
"""
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

import dishes_cv as D
from dish_boxes import DIRTY
from dish_crops import open_img
from extract_features import OUT, get_logger, image_paths, load_model, unit

log = get_logger("soiled_windows", "soiled_windows.log")
SKIP = {28, 31}
WIN = [(c * st, r * st, c * st + s, r * st + s) for s, st, n in ((1 / 3, 1 / 6, 5), (1 / 4, 1 / 8, 7))
       for r in range(n) for c in range(n)]
WIN = np.array(WIN)
PHOTOS = sorted(set(D.d_ids) | {int(i) for i in np.array(D.t_ids)[D.t_dish]})


@torch.no_grad()
def encode():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    enc, pre = load_model("dinov2_l", dev)
    paths, out = image_paths(), []
    for k, i in enumerate(PHOTOS, 1):
        im = open_img(i, paths); w, h = im.size
        crops = [im.crop((int(a * w), int(b * h), int(c * w), int(d * h))) for a, b, c, d in WIN]
        e = np.concatenate([unit(enc(torch.stack([pre(c) for c in crops[s:s + 37]]).to(dev))) for s in (0, 37)])
        out.append(e.astype(np.float32))
        if k % 10 == 0:
            log.info(f"windows {k}/{len(PHOTOS)}")
    np.savez(OUT / "soiled_windows.npz", ids=np.array(PHOTOS), boxes=WIN, emb=np.stack(out))
    log.info(f"saved features/soiled_windows.npz {np.stack(out).shape}")


def overlap(win, box):
    ix = max(0, min(win[2], box[2]) - max(win[0], box[0])); iy = max(0, min(win[3], box[3]) - max(win[1], box[1]))
    inter = ix * iy
    return inter / ((win[2] - win[0]) * (win[3] - win[1])), inter / ((box[2] - box[0]) * (box[3] - box[1]))


def window_labels(i, incomplete):
    """1 = soiled, 0 = negative, -1 = ignore, for the 74 windows of photo i."""
    if i in SKIP:
        return np.full(len(WIN), -1)
    if not incomplete:
        return np.zeros(len(WIN), int)
    lab = np.full(len(WIN), -1)
    for k, w in enumerate(WIN):
        fw, fb = overlap(w, DIRTY[i])
        lab[k] = 1 if (fw >= .6 or fb >= .6) else (0 if fw < .1 else -1)
    return lab


def main_cv():
    z = np.load(OUT / "soiled_windows.npz"); E = z["emb"]; pos = {int(i): k for k, i in enumerate(z["ids"])}
    cr = np.load(OUT / "dish_crops.npz"); cpos = {int(i): k for k, i in enumerate(cr["ids"])}
    ids, yd = np.array(D.d_ids), D.yd
    LAB = {i: window_labels(i, yd[j] == 1) for j, i in enumerate(ids)}
    ev = D.ev
    n_rep = len(D.SPLITS) // 5
    smax = np.zeros((n_rep, len(ids))); stop3 = np.zeros((n_rep, len(ids)))
    for k, (tr, va) in enumerate(D.SPLITS):
        Xs, ys = [], []
        for j in tr:
            i = int(ids[j]); m = LAB[i] >= 0
            Xs.append(E[pos[i]][m]); ys.append(LAB[i][m])
            if i in cpos and i not in SKIP:                           # hand crops as extra examples
                Xs.append(cr["feats"][cpos[i]]); ys.append(np.full(cr["feats"].shape[1], cr["y"][cpos[i]]))
        Xs, ys = np.vstack(Xs), np.concatenate(ys)
        clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(Xs, ys)
        for j in va:
            s = clf.predict_proba(E[pos[int(ids[j])]])[:, 1]
            smax[k // 5, j] = s.max(); stop3[k // 5, j] = np.sort(s)[-3:].mean()
    lines = []
    auc = [roc_auc_score(yd[ev], smax[r, ev]) for r in range(n_rep)]
    lines.append(f"soiled_max alone: AUC Incomplete vs Complete on 33 dishes photos {np.mean(auc):.3f} ± {np.std(auc):.3f}")
    order = sorted(ev, key=lambda j: smax[:, j].mean())
    lines.append("mean held-out soiled_max: " + ", ".join(f"#{ids[j]}{'I' if yd[j] else 'C'} {smax[:, j].mean():.2f}" for j in order))

    # dishes model with the scores appended (scores of training photos: from the folds where they were held out
    # would leak nothing, but simpler and stricter: recompute train scores in-fold is costly -> use held-out scores of
    # the same repeat for training photos too; they come from classifiers that never saw that photo)
    Xa, _ = D.FEATS["dinov2_l full"]; AUG = D.AUGS["dinov2_l"]
    for name, scale in (("dinov2 LR +10 aug", None), ("+ soiled x1", 1.0), ("+ soiled x3", 3.0)):
        P = np.zeros((n_rep, len(ids)))
        for k, (tr, va) in enumerate(D.SPLITS):
            r = k // 5
            F = Xa if scale is None else np.hstack([Xa, scale * np.stack([smax[r], stop3[r]], 1)])
            Ftr = F[tr]; ytr = yd[tr]
            augX = AUG[tr, :10]
            if scale is not None:
                augX = np.concatenate([augX, np.repeat(F[tr, None, -2:], 10, 1)], -1)
            Ftr = np.vstack([Ftr, augX.reshape(-1, F.shape[1])]); ytr = np.r_[ytr, np.repeat(yd[tr], 10)]
            P[r, va] = D.score(D.MODELS["LR C=1"]().fit(Ftr, ytr), F[va])
        pred = (P[:, ev] > .5).astype(int)
        f1 = [f1_score(yd[ev], p, average="macro") for p in pred]
        wrong = {int(ids[j]): int(n) for j, n in zip(ev, (pred != yd[ev]).sum(0)) if n}
        lines.append(f"{name:20s} dishes F1 {np.mean(f1):.3f} ± {np.std(f1):.3f}  wrong {wrong}")
    txt = "\n".join(lines); print(txt)
    (D.ROOT / "logs/soiled_windows_cv.txt").write_text(txt + "\n")

    # where does it fire? classifier on all training photos, maps for the hard photos
    Xs, ys = [], []
    for j, i in enumerate(ids):
        m = LAB[int(i)] >= 0; Xs.append(E[pos[int(i)]][m]); ys.append(LAB[int(i)][m])
    clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(np.vstack(Xs), np.concatenate(ys))
    paths, tiles = image_paths(), []
    for i in (2, 60, 99, 101, 46, 12):
        im = open_img(i, paths); im.thumbnail((420, 420)); w, h = im.size
        s = clf.predict_proba(E[pos[i]])[:, 1]
        dr = ImageDraw.Draw(im)
        for k in np.argsort(s)[-3:]:
            a, b, c, d = WIN[k]; dr.rectangle([a * w, b * h, c * w, d * h], outline=(255, 0, 0), width=3)
        dr.rectangle([0, 0, 200, 18], fill="black"); dr.text((4, 3), f"#{i} max {s.max():.2f} (in-sample)", fill="white")
        tiles.append(im)
    W = sum(t.width for t in tiles) + 10 * len(tiles); c = Image.new("RGB", (W, 420), "white"); x = 0
    for t in tiles:
        c.paste(t, (x, 0)); x += t.width + 10
    c.save(D.ROOT / "logs/soiled_windows_maps.jpg")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["encode", "cv"]
    if "encode" in steps:
        encode()
    if "cv" in steps:
        main_cv()
