"""Zoom-in / zoom-out augmentation for the submitted model (DINOv2 full + 4 overflow tile scores + RBF SVM).

For every training photo (data/train.csv) four extra views are made:
  in_1.25 / in_1.5   : centre crop keeping 80% / 67% of each side (zoom in)
  out_0.8 / out_0.67 : the whole photo shrunk to 80% / 67% and placed on a canvas of the original size,
                       padded with the photo's mean colour (zoom out)
Each view gets the same features as the original photo: DINOv2 image + mirror embedding, and the 4 overflow
tile scores (2x2 + 3x3 tiles of the view scored by the CDCM classifier from overflow_aux.py).
  -> features/zoom_aug.npz (ids, zooms, dinov2 (N, Z, D), tiles (N, Z, 4))

CV (same 10x5 photo-grouped splits as aux_cv.py): zoom rows of a photo are only added when that photo is in the
training fold; validation always uses the original photos only.  -> logs/zoom_cv.csv, logs/zoom_cv_photos.csv
Usage: python zoom_aug.py [encode] [cv] [--open]   (default: both; encode only adds photos missing from the file)
"""
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from extract_features import OUT, get_logger, image_paths, load_model, unit

ZOOMS = ["in_1.25", "in_1.5", "out_0.8", "out_0.67"]
BOXES = [(c / g, r / g, (c + 1) / g, (r + 1) / g) for g in (2, 3) for r in range(g) for c in range(g)]
log = get_logger("zoom_aug", "zoom_aug.log")


def zoom(im, z):
    w, h = im.size
    kind, f = z.split("_")
    f = float(f)
    if kind == "in":
        cw, ch = int(w / f), int(h / f)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return im.crop((x0, y0, x0 + cw, y0 + ch))
    small = im.resize((int(w * f), int(h * f)), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), tuple(int(v) for v in np.asarray(im).reshape(-1, 3).mean(0)))
    canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
    return canvas


def tiles_of(im):
    w, h = im.size
    return [im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))) for x0, y0, x1, y1 in BOXES]


@torch.no_grad()
def encode():
    import pandas as pd
    from overflow_aux import fit
    ids = list(pd.read_csv("data/train.csv").id)
    prev = np.load(OUT / "zoom_aug.npz") if (OUT / "zoom_aug.npz").exists() else None
    old = {int(i): k for k, i in enumerate(prev["ids"])} if prev is not None else {}
    todo = [i for i in ids if i not in old]           # only photos without zoom views are encoded
    log.info(f"{len(ids) - len(todo)} photos reused, {len(todo)} to encode: {todo}")
    paths = image_paths()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {"ids": np.array(ids), "zooms": np.array(ZOOMS)}
    tile_cols = []
    for b in ("dinov2_l", "siglip_so400m"):
        t0 = time.time()
        clf = fit(b)
        enc, pre = load_model(b, device)
        new = {}
        for k, i in enumerate(todo, 1):
            im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
            im.thumbnail((1536, 1536))
            fv, mx, t2 = [], [], []
            for z in ZOOMS:
                v = zoom(im, z)
                e = unit(enc(torch.stack([pre(v), pre(ImageOps.mirror(v))]).to(device)))
                fv.append(unit(e.sum(0)))
                s = clf.predict_proba(unit(enc(torch.stack([pre(t) for t in tiles_of(v)]).to(device))))[:, 1]
                mx.append(s.max()); t2.append(np.sort(s)[-2:].mean())
            new[i] = (fv, mx, t2)
            if k % 20 == 0:
                log.info(f"[{b}] {k}/{len(todo)}  {time.time() - t0:.0f}s")
        c = len(tile_cols)                             # column of this backbone's tile_max in prev["tiles"]
        get = lambda i: new[i] if i in new else (prev["dinov2"][old[i]] if b == "dinov2_l" else None,
                                                  prev["tiles"][old[i], :, c], prev["tiles"][old[i], :, c + 1])
        if b == "dinov2_l":
            out["dinov2"] = np.array([get(i)[0] for i in ids], dtype=np.float32)
        tile_cols += [np.array([get(i)[1] for i in ids]), np.array([get(i)[2] for i in ids])]
        if device == "mps":
            torch.mps.empty_cache()
    out["tiles"] = np.stack(tile_cols, -1)          # (N, Z, 4) same column order as overflow_tiles keys
    np.savez(OUT / "zoom_aug.npz", **out)
    log.info(f"saved features/zoom_aug.npz {out['dinov2'].shape}")


def cv():
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from aux_cv import ALPHA, BASE, MODELS, REPEATS, ROOT, ext, orig, proba, train, y

    t = np.load(ROOT / "features/overflow_tiles.npz")
    tpos = {i: k for k, i in enumerate(t["ids"])}
    TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    X = np.hstack([BASE["dinov2 full"], ALPHA * np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1)])
    zf = np.load(OUT / "zoom_aug.npz")
    assert list(zf["ids"]) == list(train.id)
    XZ = np.concatenate([zf["dinov2"], ALPHA * zf["tiles"]], -1)        # (N, Z, D + 4)
    if OPEN:   # open-bag score (bag_tiles.py), same x4 scale as aux_cv.py; zoom rows copy their photo's score
        b = np.load(ROOT / "features/bag_tiles.npz")
        bpos = {i: k for k, i in enumerate(b["ids"])}
        O = 4.0 * np.stack([b[k][[bpos[i] for i in train.id]] for k in ("open_tile_max", "open_tile_top2")], 1)
        X = np.hstack([X, ALPHA * O])
        XZ = np.concatenate([XZ, np.repeat(ALPHA * O[:, None], XZ.shape[1], 1)], -1)
    zi = {z: k for k, z in enumerate(zf["zooms"])}
    variants = {"no zoom": [], "+ zoom in": ["in_1.25", "in_1.5"], "+ zoom out": ["out_0.8", "out_0.67"],
                "+ zoom in + out": ZOOMS}
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    watch = [26, 72, 19, 63, 13, 77, 41, 2, 60, 31, 101]
    rows, photos = [], {}
    for vname, zs in variants.items():
        for mname, make in MODELS.items():
            P = np.zeros((REPEATS, len(train), 4))
            for k, (tr, va) in enumerate(splits):
                Xtr = np.vstack([X[tr]] + [XZ[tr, zi[z]] for z in zs])
                ytr = np.concatenate([y[tr]] * (1 + len(zs)))
                P[k // 5, va] = proba(make().fit(Xtr, ytr), X[va])
            f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
            rows.append({"variant": vname, "model": mname, "train rows per fold": len(splits[0][0]) * (1 + len(zs)),
                         "CV macro F1": np.mean(f1), "std": np.std(f1)})
            photos[f"{vname} | {mname}"] = {f"#{w}": f"{P[:, j, y[j]].mean():.2f} ({(P[:, j].argmax(1) != y[j]).sum()}/{REPEATS})"
                                            for w in watch for j in [int(np.where(train.id == w)[0][0])]}
            print(f"{vname:18s} | {mname:20s} {np.mean(f1):.3f} ± {np.std(f1):.3f}", flush=True)
    res = pd.DataFrame(rows)
    tag = "_open" if OPEN else ""
    res.to_csv(ROOT / f"logs/zoom_cv{tag}.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT / f"logs/zoom_cv{tag}_photos.csv")
    pd.set_option("display.width", 200)
    print("\n" + res.pivot_table(index="model", columns="variant", values="CV macro F1")[list(variants)].round(3).to_string())
    print("\n" + pd.DataFrame(photos)[[c for c in photos if "RBF SVM" in c]].to_string())


OPEN = "--open" in sys.argv       # add the open-bag score (bag_tiles.py) to every row

if __name__ == "__main__":
    steps = [a for a in sys.argv[1:] if a != "--open"] or ["encode", "cv"]
    if "encode" in steps:
        encode()
    if "cv" in steps:
        cv()
