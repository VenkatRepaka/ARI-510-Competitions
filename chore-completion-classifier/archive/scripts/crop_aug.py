"""Crop first, then transform: training-row augmentation + test-time augmentation (TTA) for the submitted model
(DINOv2 full + 4 overflow tile scores + RBF SVM).

Views per photo = CROPS x TRANSFORMS, minus (full, none) which is the original row:
  CROPS      : full, and four off-centre 1.5x zooms (crop keeping 67% of each side at the tl / tr / bl / br corner)
  TRANSFORMS : none, grey (greyscale), dark (gamma 2.8 + light noise), rot (+12 deg on tl/br/full, -12 deg on tr/bl,
               then the largest same-aspect centre crop, so there are no empty corners)
  -> 5 x 4 - 1 = 19 views. Each view gets the same features as the original photo: DINOv2 image + mirror embedding
     and the 4 overflow tile scores (2x2 + 3x3 tiles scored by the CDCM classifier from overflow_aux.py).
  -> features/crop_aug_dinov2_l.npz, features/crop_aug_siglip_so400m.npz  (one per backbone, so a crash keeps the first)

CV (same 10x5 photo-grouped splits as aux_cv.py): augmented rows of a photo are only added when that photo is in the
training fold. Validation uses the original photos; with TTA the predicted probabilities of the original and of some of
its views are averaged.  -> logs/crop_aug_cv.csv, logs/crop_aug_cv_photos.csv
Usage: python crop_aug.py [encode] [cv]   (default: both)
"""
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from extract_features import OUT, get_logger, image_paths, load_model, unit
from zoom_aug import tiles_of

CROPS = ["full", "tl", "tr", "bl", "br"]
TRANSFORMS = ["none", "grey", "dark", "rot"]
VIEWS = [(c, t) for c in CROPS for t in TRANSFORMS if (c, t) != ("full", "none")]
NAMES = [f"{c}+{t}" for c, t in VIEWS]
BACKBONES = ("dinov2_l", "siglip_so400m")
log = get_logger("crop_aug", "crop_aug.log")


def crop(im, c):
    if c == "full":
        return im
    w, h = im.size
    cw, ch = int(w / 1.5), int(h / 1.5)
    x0 = 0 if c[1] == "l" else w - cw
    y0 = 0 if c[0] == "t" else h - ch
    return im.crop((x0, y0, x0 + cw, y0 + ch))


def rotate(im, deg):
    w, h = im.size
    th = np.radians(abs(deg))
    s = 1 / (np.cos(th) + max(w, h) / min(w, h) * np.sin(th))   # largest same-aspect rectangle inside the rotated photo
    cw, ch = int(w * s), int(h * s)
    return im.rotate(deg, resample=Image.BICUBIC).crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))


def transform(im, c, t, seed):
    if t == "none":
        return im
    if t == "grey":
        return im.convert("L").convert("RGB")
    if t == "dark":
        a = 255 * (np.asarray(im, dtype=np.float32) / 255) ** 2.8
        a += np.random.default_rng(seed).normal(0, 5, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return rotate(im, -12 if c in ("tr", "bl") else 12)


def views(im, seed):
    return [transform(crop(im, c), c, t, seed * 10 + k) for k, (c, t) in enumerate(VIEWS)]


def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((1536, 1536))
    return im


@torch.no_grad()
def encode():
    import pandas as pd
    from overflow_aux import fit
    ids = list(pd.read_csv("data/train.csv").id)
    paths = image_paths()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for b in BACKBONES:
        path = OUT / f"crop_aug_{b}.npz"
        if path.exists():
            log.info(f"[{b}] {path.name} exists, skipping")
            continue
        t0 = time.time()
        clf = fit(b)
        enc, pre = load_model(b, device)
        full, tmax, ttop2 = [], [], []
        for k, i in enumerate(ids, 1):
            fv, mx, t2 = [], [], []
            for v in views(load(paths[i]), i):
                if b == "dinov2_l":
                    e = unit(enc(torch.stack([pre(v), pre(ImageOps.mirror(v))]).to(device)))
                    fv.append(unit(e.sum(0)))
                s = clf.predict_proba(unit(enc(torch.stack([pre(t) for t in tiles_of(v)]).to(device))))[:, 1]
                mx.append(s.max()); t2.append(np.sort(s)[-2:].mean())
            full.append(fv); tmax.append(mx); ttop2.append(t2)
            if k % 10 == 0:
                el = time.time() - t0
                log.info(f"[{b}] {k}/{len(ids)}  {el:.0f}s  (~{el / k * (len(ids) - k) / 60:.0f} min left for this backbone)")
        out = dict(ids=np.array(ids), views=np.array(NAMES), tile_max=np.array(tmax), tile_top2=np.array(ttop2))
        if b == "dinov2_l":
            out["full"] = np.array(full, dtype=np.float32)
        np.savez(path, **out)
        log.info(f"saved {path.name}")
        del enc
        if device == "mps":
            torch.mps.empty_cache()


def cv():
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from aux_cv import ALPHA, BASE, MODELS, REPEATS, ROOT, ext, orig, proba, train, y

    t = np.load(ROOT / "features/overflow_tiles.npz")
    tpos = {i: k for k, i in enumerate(t["ids"])}
    TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    X = np.hstack([BASE["dinov2 full"], ALPHA * np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1)])
    d, s = (np.load(OUT / f"crop_aug_{b}.npz") for b in BACKBONES)
    assert list(d["ids"]) == list(train.id) == list(s["ids"]) and list(d["views"]) == NAMES
    XV = np.concatenate([d["full"], ALPHA * np.stack([d["tile_max"], d["tile_top2"], s["tile_max"], s["tile_top2"]], -1)], -1)
    vi = {n: k for k, n in enumerate(NAMES)}

    corners = [f"{c}+none" for c in CROPS[1:]]
    colour = ["full+grey", "full+dark", "full+rot"]
    train_variants = {"no aug": [], "+ crops": corners, "+ grey": ["full+grey"], "+ dark": ["full+dark"],
                      "+ rot": ["full+rot"], "+ grey/dark/rot": colour, "+ crops then all (19)": NAMES}
    tta = {"no TTA": [], "TTA grey/dark/rot": colour, "TTA crops": corners, "TTA all (19)": NAMES}
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    watch = [26, 72, 19, 63, 13, 77, 41, 2, 60, 31, 101]
    wj = {w: int(np.where(train.id == w)[0][0]) for w in watch}
    rows, photos = [], {}
    for vname, zs in train_variants.items():
        for mname, make in MODELS.items():
            t0 = time.time()
            P = {k: np.zeros((REPEATS, len(train), 4)) for k in tta}
            for r, (tr, va) in enumerate(splits):
                Xtr = np.vstack([X[tr]] + [XV[tr, vi[z]] for z in zs])
                ytr = np.concatenate([y[tr]] * (1 + len(zs)))
                m = make().fit(Xtr, ytr)
                p0 = proba(m, X[va])
                pv = {n: proba(m, XV[va, vi[n]]) for n in NAMES}
                for k, ns in tta.items():
                    P[k][r // 5, va] = np.mean([p0] + [pv[n] for n in ns], 0)
            for k, Pk in P.items():
                f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in Pk]
                rows.append({"train rows": vname, "TTA": k, "model": mname, "CV macro F1": np.mean(f1), "std": np.std(f1)})
                photos[f"{vname} | {k} | {mname}"] = {
                    f"#{w}": f"{Pk[:, j, y[j]].mean():.2f} ({(Pk[:, j].argmax(1) != y[j]).sum()}/{REPEATS})" for w, j in wj.items()}
                print(f"{vname:22s} | {k:18s} | {mname:20s} {np.mean(f1):.3f} ± {np.std(f1):.3f}", flush=True)
            log.info(f"cv {vname} | {mname} done in {time.time() - t0:.0f}s")
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "logs/crop_aug_cv.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT / "logs/crop_aug_cv_photos.csv")
    pd.set_option("display.width", 250)
    for mname in MODELS:
        print(f"\n{mname}\n" + res[res.model == mname].pivot_table(index="train rows", columns="TTA", values="CV macro F1")
              .loc[list(train_variants), list(tta)].round(3).to_string())
    print("\nRBF SVM, per photo: mean p(true label) (times wrong out of 10)")
    print(pd.DataFrame(photos)[[f"{v} | {k} | RBF SVM" for v in train_variants for k in ("no TTA", "TTA all (19)")]].T.to_string())


if __name__ == "__main__":
    steps = sys.argv[1:] or ["encode", "cv"]
    if "encode" in steps:
        encode()
    if "cv" in steps:
        cv()
