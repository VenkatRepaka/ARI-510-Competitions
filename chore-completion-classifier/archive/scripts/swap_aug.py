"""Crop + orientation + colour-swap augmentation, trained on the 66 competition photos only (no external photos).

Views made from every competition training photo (13 per photo):
  crops   (4): 60% of the photo kept from one side, each turned a different way so none is upright:
               bottom 60% -> rotate 90 clockwise, top 60% -> rotate 180,
               left 60%   -> rotate 90 anticlockwise, right 60% -> flip top-to-bottom
  crops+swap (4): the same 4 crops with the colour swap below
  full    (4): the whole photo with the same 4 orientation changes (nothing cut away)
  full+swap (1): the whole photo, upright, colour-swapped
Colour swap: hue mapping red <-> blue, green <-> yellow, cyan <-> magenta, hues in between interpolated
along the shorter way round the colour circle; saturation and brightness unchanged.
Each view and its mirror are encoded with DINOv2 (same preprocessing as extract_features.py) and averaged,
like the original rows.  -> features/swap_aug_dinov2_l.npz (ids, views (66, 13, D), names)

Label risk: a 60% crop can cut away the mess, so a crop of an Incomplete photo can look Complete while keeping the
Incomplete label. The CV therefore compares:
  no aug                                : 66 photos only
  + full orientations (4)               : nothing cut away, label always still right
  + full orientations + colour (5)      : same, plus the colour swap
  + crops, all photos (8)               : the crops as proposed, for every photo
  + crops for Complete only (8) + full (5): crops only where cutting cannot remove a mess
  + everything (13)
It also measures the risk directly: the no-aug model (trained without the photo) predicts each crop of a held-out
photo; reported is how often a crop of an Incomplete photo is predicted Complete (and the reverse).

Protocol: 10 x 5-fold photo-grouped CV (same splits as aux_cv.py); a photo's views are only in training when the
photo is; only the 66 original photos are scored. Two feature sets:
  dinov2                : DINOv2 only (strictly no external data)
  dinov2 + tiles + open : the submitted feature set; tile and open-bag scores of a view are copied from its photo.
                          These scores come from classifiers/references built on external images (CDCM, bag photos),
                          so this set is not strictly external-free.
-> logs/swap_aug_cv.csv, logs/swap_aug_photos.csv, logs/swap_aug_flip.csv, logs/swap_aug_views_sheet.jpg
Usage: python swap_aug.py [encode] [cv]   (default both)
Reference: Shorten & Khoshgoftaar 2019, "A survey on Image Data Augmentation for Deep Learning", J. Big Data.
"""
import os
import sys
import time

os.environ.setdefault("TRAIN_CSV", "data/train_original.csv")   # 66 competition photos, no externals

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps

from extract_features import OUT, get_logger, image_paths, load_model, unit

ROOT_DIR = OUT.parent
log = get_logger("swap_aug", "swap_aug.log")
T = Image.Transpose
CROP_ORIENT = [("bottom", T.ROTATE_270, "rot90cw"), ("top", T.ROTATE_180, "rot180"),
               ("left", T.ROTATE_90, "rot90ccw"), ("right", T.FLIP_TOP_BOTTOM, "flipV")]
FRAC = 0.60

# colour swap lookup table on the PIL hue channel (0-255 = 0-360 degrees)
_src = np.array([0, 60, 120, 180, 240, 300, 360], float)
_dst = np.array([240, 120, 60, 300, 0, 180, 240], float)
for _k in range(1, len(_dst)):
    _dst[_k] = _dst[_k - 1] + (_dst[_k] - _dst[_k - 1] + 180) % 360 - 180
HUE_LUT = (np.interp(np.arange(256) * 360 / 256, _src, _dst) % 360 * 256 / 360).astype(np.uint8).tolist()


def color_swap(im):
    h, s, v = im.convert("HSV").split()
    return Image.merge("HSV", (h.point(HUE_LUT), s, v)).convert("RGB")


def crop_side(im, side):
    w, h = im.size
    return im.crop({"bottom": (0, int(h * (1 - FRAC)), w, h), "top": (0, 0, w, int(h * FRAC)),
                    "left": (0, 0, int(w * FRAC), h), "right": (int(w * (1 - FRAC)), 0, w, h)}[side])


def make_views(im):
    crops = [(f"crop_{s}_{n}", crop_side(im, s).transpose(t)) for s, t, n in CROP_ORIENT]
    full = [(f"full_{n}", im.transpose(t)) for _, t, n in CROP_ORIENT]
    return (crops + [(k + "_swap", color_swap(v)) for k, v in crops] + full
            + [("full_upright_swap", color_swap(im))])


VIEW_NAMES = [k for k, _ in make_views(Image.new("RGB", (8, 6)))]


@torch.no_grad()
def encode():
    ids = list(pd.read_csv(ROOT_DIR / "data/train_original.csv").id)
    path = OUT / "swap_aug_dinov2_l.npz"
    if path.exists() and list(np.load(path)["ids"]) == ids:
        log.info("features/swap_aug_dinov2_l.npz already has every photo, skipping")
        return
    paths = image_paths()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc, pre = load_model("dinov2_l", device)
    out, t0 = [], time.time()
    for k, i in enumerate(ids, 1):
        im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
        im.thumbnail((1536, 1536))
        vs = [v for _, v in make_views(im)]
        e = unit(enc(torch.stack([pre(v) for v in vs] + [pre(ImageOps.mirror(v)) for v in vs]).to(device)))
        out.append(unit(e[:len(vs)] + e[len(vs):]).astype(np.float32))
        if i == 26:   # one example sheet so the views can be checked by eye
            sheet(vs)
        if k % 10 == 0:
            log.info(f"{k}/{len(ids)}  {time.time() - t0:.0f}s")
        if device == "mps":
            torch.mps.empty_cache()
    np.savez(path, ids=np.array(ids), views=np.stack(out), names=np.array(VIEW_NAMES))
    log.info(f"saved {path.name} {np.stack(out).shape} in {time.time() - t0:.0f}s")


def sheet(vs, H=200):
    tiles = []
    for name, v in zip(VIEW_NAMES, vs):
        t = v.resize((max(1, int(v.width * H / v.height)), H))
        x = Image.new("RGB", (t.width, H + 18), "white")
        x.paste(t, (0, 18)); ImageDraw.Draw(x).text((3, 3), name, fill="black")
        tiles.append(x)
    rows = [tiles[:4], tiles[4:8], tiles[8:]]
    W = max(sum(t.width for t in r) + 8 * len(r) for r in rows)
    s = Image.new("RGB", (W, (H + 26) * 3), "white")
    for r_i, r in enumerate(rows):
        xo = 0
        for t in r:
            s.paste(t, (xo, r_i * (H + 26))); xo += t.width + 8
    s.save(ROOT_DIR / "logs/swap_aug_views_sheet.jpg", quality=85)


def cv():
    from sklearn.metrics import f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from aux_cv import ALPHA, BASE, MODELS, REPEATS, orig, proba, train, y

    assert len(train) == 66, "expects TRAIN_CSV=data/train_original.csv"
    r = np.load(OUT / "swap_aug_dinov2_l.npz")
    assert list(r["ids"]) == list(train.id)
    V = r["views"]                                                      # (66, 13, D)
    names = list(r["names"])
    idx = lambda pre: [k for k, n in enumerate(names) if n.startswith(pre)]
    CROPS, FULL = idx("crop_"), idx("full_")
    FULL_ORIENT = [k for k in FULL if not names[k].endswith("swap")]

    t = np.load(ROOT_DIR / "features/overflow_tiles.npz"); tpos = {i: k for k, i in enumerate(t["ids"])}
    b = np.load(ROOT_DIR / "features/bag_tiles.npz"); bpos = {i: k for k, i in enumerate(b["ids"])}
    TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    EXTRA = np.hstack([ALPHA * np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1),
                       ALPHA * 4.0 * np.stack([b[k][[bpos[i] for i in train.id]] for k in ("open_tile_max", "open_tile_top2")], 1)])
    feats = {"dinov2": (BASE["dinov2 full"], V),
             "dinov2 + tiles + open": (np.hstack([BASE["dinov2 full"], EXTRA]),
                                       np.concatenate([V, np.repeat(EXTRA[:, None], V.shape[1], 1)], -1))}
    complete = np.isin(y, [0, 2])
    variants = {"no aug": lambda j: [],
                "+ full orientations (4)": lambda j: FULL_ORIENT,
                "+ full orientations + colour (5)": lambda j: FULL,
                "+ crops, all photos (8)": lambda j: CROPS,
                "+ crops Complete only (8) + full (5)": lambda j: (CROPS if complete[j] else []) + FULL,
                "+ everything (13)": lambda j: CROPS + FULL}
    splits = [(orig[tr], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    watch = [26, 72, 19, 63, 13, 77, 41, 2, 60, 31, 101, 99]
    wj = {w: int(np.where(train.id == w)[0][0]) for w in watch}
    dsh, trs = orig[y[orig] < 2], orig[y[orig] >= 2]
    rows, photos, flips = [], {}, []
    for fname, (X, A) in feats.items():
        for vname, pick in variants.items():
            for mname, make in MODELS.items():
                t0 = time.time()
                P = np.zeros((REPEATS, len(train), 4))
                for s, (tr, va) in enumerate(splits):
                    extra = [(A[j, pick(j)], np.full(len(pick(j)), y[j])) for j in tr if pick(j)]
                    Xtr = np.vstack([X[tr]] + [e[0] for e in extra])
                    ytr = np.concatenate([y[tr]] + [e[1] for e in extra])
                    m = make().fit(Xtr, ytr)
                    P[s // 5, va] = proba(m, X[va])
                    if vname == "no aug":        # label-risk check: does the no-aug model see the crops as the same class?
                        pc = proba(m, A[va][:, CROPS].reshape(-1, A.shape[-1])).argmax(1).reshape(len(va), -1)
                        for j, p in zip(va, pc):
                            flips.append({"features": fname, "model": mname, "id": int(train.id[j]), "label": y[j],
                                          "photo predicted right": int(P[s // 5, j].argmax() == y[j]),
                                          "crops predicted right": float((p == y[j]).mean()),
                                          "crops flipped Complete<->Incomplete": float((p == (y[j] ^ 1)).mean())})
                f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
                f1_d = [f1_score(y[dsh], p[dsh].argmax(1), labels=[0, 1], average="macro") for p in P]
                f1_t = [f1_score(y[trs], p[trs].argmax(1), labels=[2, 3], average="macro") for p in P]
                rows.append({"features": fname, "train rows": vname, "model": mname, "CV macro F1": np.mean(f1),
                             "std": np.std(f1), "dishes F1": np.mean(f1_d), "trash F1": np.mean(f1_t)})
                photos[f"{fname} | {vname} | {mname}"] = {
                    f"#{w}": f"{P[:, j, y[j]].mean():.2f} ({(P[:, j].argmax(1) != y[j]).sum()}/{REPEATS})" for w, j in wj.items()}
                print(f"{fname:22s} | {vname:38s} | {mname:20s} {np.mean(f1):.3f} ± {np.std(f1):.3f}  "
                      f"dishes {np.mean(f1_d):.3f} trash {np.mean(f1_t):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT_DIR / "logs/swap_aug_cv.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT_DIR / "logs/swap_aug_photos.csv")
    fl = pd.DataFrame(flips)
    fl.to_csv(ROOT_DIR / "logs/swap_aug_flip.csv", index=False)
    pd.set_option("display.width", 250)
    for fname in feats:
        print(f"\n== {fname}: CV macro F1 (66 photos, no externals) ==")
        print(res[res.features == fname].pivot_table(index="model", columns="train rows", values="CV macro F1")
              [list(variants)].round(3).to_string())
    cls = {0: "Dishes Complete", 1: "Dishes Incomplete", 2: "Trash Complete", 3: "Trash Incomplete"}
    print("\nLabel risk (no-aug model, crops of held-out photos): share of crops predicted as the OTHER completion state")
    print(fl.assign(cls=fl.label.map(cls)).pivot_table(index=["features", "model"], columns="cls",
          values="crops flipped Complete<->Incomplete").round(2).to_string())


if __name__ == "__main__":
    steps = sys.argv[1:] or ["encode", "cv"]
    if "encode" in steps:
        encode()
    if "cv" in steps:
        cv()
