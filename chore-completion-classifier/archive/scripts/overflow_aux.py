"""Auxiliary "dirty / overflowing bin" classifier trained on an external dataset, used as an extra feature.

Data: Clean-Dirty Containers in Montevideo (CDCM) v6.1, Kaggle rodrigolaguna/clean-dirty-containers-in-montevideo,
unzipped by hand into ../external_images/cdcm/. Labels: clean / dirty street garbage containers.
Only the "smartphone" source is used: other sources are confounded with the label (por_mi_barrio ~ all dirty,
Google Street View ~ mostly clean), so a classifier could learn the source instead of the dirt.

Steps
  encode  : sample N_TRAIN per class from CDCM train and N_VAL per class from CDCM test (seeded), encode image + mirror
            with each backbone using the same preprocessing as extract_features.py -> features/cdcm_<backbone>.npz
  train   : logistic regression on CDCM train embeddings, report accuracy / AUC on the CDCM validation sample
  score   : apply it to every competition and external photo, using the features already in features/<backbone>.npz,
            -> features/overflow_score.npz with p_dirty on the full view (image + mirror) and the max over the 7 views

The CDCM images are only used to train this small classifier; they never enter the chore training set.
  tiles   : zoom: score a 2x2 + 3x3 grid of tiles of every photo -> features/overflow_tiles.npz (see tiles())

Usage: python overflow_aux.py [encode] [train] [score] [tiles]   (default: encode, train, score)
"""
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from extract_features import OUT, get_logger, load_model, unit

ROOT = Path(__file__).parent
CDCM = ROOT.parent / "external_images/cdcm/images/clean-dirty-garbage-containers-V6.1"
BACKBONES = ["dinov2_l", "siglip_so400m"]
N_TRAIN, N_VAL, SEED = 300, 100, 0
log = get_logger("overflow_aux", "overflow_aux.log")


def sample():
    m = pd.read_csv(CDCM / "clean-dirty-garbage-containers/clean-dirty-metadata.csv", sep=";")
    m = m[m.source == "smartphone"]
    parts = []
    for split, n in (("train", N_TRAIN), ("test", N_VAL)):
        for label in ("clean", "dirty"):
            parts.append(m[(m.split == split) & (m.label == label)].sample(n, random_state=SEED))
    s = pd.concat(parts).reset_index(drop=True)
    s["y"] = (s.label == "dirty").astype(int)
    return s


@torch.no_grad()
def encode():
    s = sample()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for b in BACKBONES:
        t0 = time.time()
        enc, pre = load_model(b, device)
        feats = []
        for k, p in enumerate(s.image_path, 1):
            im = ImageOps.exif_transpose(Image.open(CDCM / p)).convert("RGB")
            im.thumbnail((1536, 1536))
            v = unit(enc(torch.stack([pre(im), pre(ImageOps.mirror(im))]).to(device)))
            feats.append(unit(v.sum(0)))
            if k % 100 == 0:
                log.info(f"[{b}] {k}/{len(s)}  {time.time() - t0:.0f}s")
        np.savez(OUT / f"cdcm_{b}.npz", feats=np.array(feats), y=s.y.values, split=s.split.values,
                 path=s.image_path.values)
        log.info(f"[{b}] saved features/cdcm_{b}.npz ({len(s)} images) in {time.time() - t0:.0f}s")
        if device == "mps":
            torch.mps.empty_cache()


def fit(b):
    z = np.load(OUT / f"cdcm_{b}.npz", allow_pickle=True)
    tr, va = z["split"] == "train", z["split"] == "test"
    clf = LogisticRegression(C=1, max_iter=5000).fit(z["feats"][tr], z["y"][tr])
    p = clf.predict_proba(z["feats"][va])[:, 1]
    log.info(f"[{b}] CDCM validation ({va.sum()} images): accuracy {accuracy_score(z['y'][va], p > .5):.3f}, "
             f"AUC {roc_auc_score(z['y'][va], p):.3f}")
    # refit on train + validation for scoring the competition photos
    return LogisticRegression(C=1, max_iter=5000).fit(z["feats"], z["y"])


def score():
    ids = None
    out = {}
    for b in BACKBONES:
        clf = fit(b)
        z = np.load(OUT / f"{b}.npz")
        if ids is None:
            ids = np.sort(z["ids"])
        idx = [{i: k for k, i in enumerate(z["ids"])}[i] for i in ids]      # same photo order for every backbone
        views = np.concatenate([z["img"][:, None], z["img_flip"][:, None], z["crops"]], 1)[idx]   # (N, 7, D)
        full = unit(views[:, 0] + views[:, 1])
        out[f"{b}_full"] = clf.predict_proba(full)[:, 1]
        out[f"{b}_max7"] = clf.predict_proba(views.reshape(-1, views.shape[-1]))[:, 1].reshape(len(views), 7).max(1)
        if b == "dinov2_l":   # duplicate check: no CDCM image may be (nearly) identical to a competition photo
            c = np.load(OUT / f"cdcm_{b}.npz", allow_pickle=True)["feats"]
            log.info(f"[{b}] max cosine between a CDCM image and any competition/external photo: {(full @ c.T).max():.2f}")
    out["ids"] = ids
    np.savez(OUT / "overflow_score.npz", **out)
    log.info(f"saved features/overflow_score.npz: {[k for k in out if k != 'ids']}")


def tiles():
    """Zoom without hand-picking: cut every photo (train.csv + test.csv) into a 2x2 and a 3x3 grid (13 tiles),
    score each tile with the overflow classifier, keep per-photo summaries.
    -> features/overflow_tiles.npz with ids, <backbone>_tiles (N, 13), <backbone>_tile_max, <backbone>_tile_top2."""
    from extract_features import image_paths
    ids = sorted(set(pd.read_csv(ROOT / os.environ.get("TRAIN_CSV", "data/train.csv")).id) | set(pd.read_csv(ROOT / "data/test.csv").id))
    paths = image_paths()
    boxes = [(c / g, r / g, (c + 1) / g, (r + 1) / g) for g in (2, 3) for r in range(g) for c in range(g)]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {"ids": np.array(ids)}
    prev = np.load(OUT / "overflow_tiles.npz") if (OUT / "overflow_tiles.npz").exists() else None
    for b in BACKBONES:
        t0 = time.time()
        # reuse tile scores already on disk (the CDCM classifier is deterministic); score only new photos
        have = {int(i): r for i, r in zip(prev["ids"], prev[f"{b}_tiles"])} if prev is not None else {}
        log.info(f"[{b} tiles] {sum(i in have for i in ids)} photos reused, {sum(i not in have for i in ids)} to score")
        clf = fit(b)
        enc, pre = load_model(b, device)
        scores = []
        with torch.no_grad():
            for k, i in enumerate(ids, 1):
                if i in have:
                    scores.append(have[i])
                    continue
                im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
                im.thumbnail((1536, 1536))
                w, h = im.size
                ts = [im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))) for x0, y0, x1, y1 in boxes]
                e = unit(enc(torch.stack([pre(t) for t in ts]).to(device)))
                scores.append(clf.predict_proba(e)[:, 1])
                if k % 25 == 0:
                    log.info(f"[{b} tiles] {k}/{len(ids)}  {time.time() - t0:.0f}s")
        s = np.array(scores)
        out[f"{b}_tiles"] = s
        out[f"{b}_tile_max"] = s.max(1)
        out[f"{b}_tile_top2"] = np.sort(s, 1)[:, -2:].mean(1)
        log.info(f"[{b}] tiles scored for {len(ids)} photos in {time.time() - t0:.0f}s")
        if device == "mps":
            torch.mps.empty_cache()
    np.savez(OUT / "overflow_tiles.npz", **out)
    log.info("saved features/overflow_tiles.npz")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["encode", "train", "score"]
    if "tiles" in steps:
        tiles()
    if "encode" in steps:
        encode()
    if "train" in steps:
        for b in BACKBONES:
            fit(b)
    if "score" in steps:
        score()
