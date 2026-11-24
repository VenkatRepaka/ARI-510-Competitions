"""Does the top of the photo mislead the model? Remove the top 30% and use the rest instead of the full photo.

Question (user's): in #2 the top of the photo is full of clean hanging pots and pans, and in #60 a rack of clean-looking
dishes fills the upper middle, while the actual mess (bowl of leftovers, plate in sink, pan left out) is lower down.
Maybe the clean things at the top outweigh the mess.

To keep the test fair the same crop is applied to EVERY photo (training, external and test), never to chosen ones:
  bottom70 : the photo with its top 30% cut off, DINOv2-encoded (image + mirror averaged, same preprocessing as
             extract_features.py) -> features/dinov2_l_bottom70.npz
Variants (DINOv2 part of the feature vector; tile and open-bag scores stay from the full photo):
  full            : the current best model (DINOv2 + tiles + open, RBF SVM)
  bottom70        : bottom-70% embedding instead of the full-photo embedding
  full + bottom70 : average of the two embeddings
  full | bottom70 : both embeddings side by side
Training set: TRAIN_CSV (default data/train_newbatch_clean_bins.csv). Same 10 x 5 photo-grouped CV as aux_cv.py.
-> logs/crop_top_cv.csv, logs/crop_top_photos.csv, logs/crop_top_examples.jpg
Usage: python crop_top_cv.py
"""
import os

os.environ.setdefault("TRAIN_CSV", "data/train_newbatch_clean_bins.csv")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

import aux_cv as A
from extract_features import OUT, image_paths, load_model, unit

CUT = 0.30
ids = A.train.id.values
y, orig, ext = A.y, A.orig, A.ext


def bottom(im):
    w, h = im.size
    return im.crop((0, int(h * CUT), w, h))


@torch.no_grad()
def encode():
    want = sorted(set(ids) | set(pd.read_csv(A.ROOT / "data/test.csv").id))
    path = OUT / "dinov2_l_bottom70.npz"
    have = {}
    if path.exists():
        z = np.load(path); have = {int(i): v for i, v in zip(z["ids"], z["emb"])}
    todo = [i for i in want if i not in have]
    if todo:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        enc, pre = load_model("dinov2_l", device)
        paths = image_paths()
        for k, i in enumerate(todo, 1):
            im = bottom(ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB"))
            e = unit(enc(torch.stack([pre(im), pre(ImageOps.mirror(im))]).to(device)))
            have[i] = unit(e[0] + e[1]).astype(np.float32)
            if k % 25 == 0:
                print(f"encoded {k}/{len(todo)}", flush=True)
        keep = sorted(have)
        np.savez(path, ids=np.array(keep), emb=np.stack([have[i] for i in keep]))
    return np.stack([have[int(i)] for i in ids])


def extras():
    t = np.load(OUT / "overflow_tiles.npz"); tp = {i: k for k, i in enumerate(t["ids"])}
    b = np.load(OUT / "bag_tiles.npz"); bp = {i: k for k, i in enumerate(b["ids"])}
    TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    return np.hstack([A.ALPHA * np.stack([t[k][[tp[i] for i in ids]] for k in TK], 1),
                      A.ALPHA * 4.0 * np.stack([b[k][[bp[i] for i in ids]] for k in ("open_tile_max", "open_tile_top2")], 1)])


def examples():
    paths = image_paths(); tiles = []
    for i in (2, 60, 63, 26):
        im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB"); im.thumbnail((420, 340))
        d = ImageDraw.Draw(im); w, h = im.size
        d.rectangle([0, 0, w, int(h * CUT)], outline="red", width=4); d.line([0, 0, w, int(h * CUT)], fill="red", width=3)
        c = Image.new("RGB", (420, 365), "white"); c.paste(im, (0, 25)); ImageDraw.Draw(c).text((4, 6), f"#{i}: red part removed", fill="black")
        tiles.append(c)
    s = Image.new("RGB", (430 * 4, 365), "white")
    for k, t in enumerate(tiles): s.paste(t, (k * 430, 0))
    s.save(A.ROOT / "logs/crop_top_examples.jpg", quality=85)


def main():
    full, bot, ex = A.BASE["dinov2 full"], encode(), extras()
    variants = {"full": np.hstack([full, ex]), "bottom70": np.hstack([bot, ex]),
                "full + bottom70": np.hstack([unit(full + bot), ex]), "full | bottom70": np.hstack([full, bot, ex])}
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig])]
    dsh, trs = orig[y[orig] < 2], orig[y[orig] >= 2]
    rows, wrong = [], {}
    for v, X in variants.items():
        for m in ("RBF SVM", "logistic regression"):
            P = np.zeros((A.REPEATS, len(y)), int)
            for k, (tr, va) in enumerate(splits):
                P[k // 5, va] = A.proba(A.MODELS[m]().fit(X[tr], y[tr]), X[va]).argmax(1)
            f = [f1_score(y[orig], p[orig], average="macro") for p in P]
            fd = [f1_score(y[dsh], p[dsh], labels=[0, 1], average="macro") for p in P]
            ft = [f1_score(y[trs], p[trs], labels=[2, 3], average="macro") for p in P]
            rows.append({"variant": v, "model": m, "CV macro F1": np.mean(f), "std": np.std(f),
                         "dishes F1": np.mean(fd), "trash F1": np.mean(ft)})
            wrong[f"{v} | {m}"] = {f"#{ids[j]}": int((P[:, j] != y[j]).sum()) for j in orig}
            print(f"{v:18s} | {m:20s} {np.mean(f):.3f} ± {np.std(f):.3f}  dishes {np.mean(fd):.3f}  trash {np.mean(ft):.3f}", flush=True)
    res = pd.DataFrame(rows); res.to_csv(A.ROOT / "logs/crop_top_cv.csv", index=False)
    w = pd.DataFrame(wrong); w = w[(w > 0).any(axis=1)]; w.to_csv(A.ROOT / "logs/crop_top_photos.csv")
    pd.set_option("display.width", 220)
    print("\nPhotos wrong at least once, RBF SVM (times wrong out of 10):")
    print(w[[c for c in w.columns if "SVM" in c]].to_string())
    examples()


if __name__ == "__main__":
    main()
