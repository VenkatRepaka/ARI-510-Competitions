"""60 random augmented views per training photo, encoded with DINOv2 only, used as extra training rows.

Each view (seeded per photo, so it is reproducible) is:
  random crop      : 50-100% of the photo's area, aspect 3/4-4/3, any position
  random rotation  : -12..+12 deg (half of the views), then the largest same-aspect centre crop (no empty corners)
  random greyscale : 25% of the views
  random darkening : 30% of the views (only if not already dark: mean grey > 80), gamma 1.8-3.0 + light noise
Both the view and its mirror are encoded (DINOv2, same preprocessing as extract_features.py) and saved separately,
so a training row can be "view + mirror averaged" (like the original rows) or each one on its own.
The 4 overflow tile scores of a view are copied from its original photo (recomputing them per view would cost ~13x more).
  -> features/rand_aug_dinov2_l.npz (ids, img (N, K, D), mirror (N, K, D))

CV (same 10x5 photo-grouped splits as aux_cv.py): views of a photo are only added when that photo is in the training
fold; the dev (validation) fold is scored on original photos only, optionally with TTA (mean probability over the
original and its views).  -> logs/rand_aug_cv.csv, logs/rand_aug_cv_photos.csv
Usage: python rand_aug.py [encode] [cv] [--open]   (default: both; encode only adds photos missing from the file)
"""
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from crop_aug import load, rotate
from extract_features import OUT, get_logger, image_paths, load_model, unit

K = 60
log = get_logger("rand_aug", "rand_aug.log")


def random_view(im, rng):
    w, h = im.size
    for _ in range(20):
        area = rng.uniform(0.5, 1.0) * w * h
        ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        cw, ch = int(round(np.sqrt(area * ar))), int(round(np.sqrt(area / ar)))
        if cw <= w and ch <= h:
            break
    else:
        cw, ch = w, h
    x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    v = im.crop((x0, y0, x0 + cw, y0 + ch))
    if rng.random() < 0.5:
        v = rotate(v, rng.uniform(-12, 12))
    if rng.random() < 0.25:
        v = v.convert("L").convert("RGB")
    if rng.random() < 0.3 and np.asarray(v.convert("L")).mean() > 80:    # never darken an already dark photo
        a = 255 * (np.asarray(v, dtype=np.float32) / 255) ** rng.uniform(1.8, 3.0)
        a += rng.normal(0, 5, a.shape)
        v = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return v


def views(im, seed):
    rng = np.random.default_rng(seed)
    return [random_view(im, rng) for _ in range(K)]


@torch.no_grad()
def encode():
    import pandas as pd
    ids = list(pd.read_csv("data/train.csv").id)
    paths = image_paths()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc, pre = load_model("dinov2_l", device)
    prev = np.load(OUT / "rand_aug_dinov2_l.npz") if (OUT / "rand_aug_dinov2_l.npz").exists() else None
    old = {int(i): k for k, i in enumerate(prev["ids"])} if prev is not None else {}
    todo = [i for i in ids if i not in old]           # views are seeded per photo, so reusing old rows is exact
    log.info(f"{len(ids) - len(todo)} photos reused, {len(todo)} to encode: {todo}")
    t0 = time.time()
    new = {}
    for k, i in enumerate(todo, 1):
        vs = views(load(paths[i]), i)
        a, b = [], []
        for s in range(0, K, 10):                       # batches of 10 views + their 10 mirrors
            chunk = vs[s:s + 10]
            e = unit(enc(torch.stack([pre(v) for v in chunk] + [pre(ImageOps.mirror(v)) for v in chunk]).to(device)))
            a.append(e[:len(chunk)]); b.append(e[len(chunk):])
        new[i] = (np.concatenate(a), np.concatenate(b))
        if k % 10 == 0:
            el = time.time() - t0
            log.info(f"{k}/{len(todo)}  {el:.0f}s  (~{el / k * (len(todo) - k) / 60:.0f} min left)")
    get = lambda i: new[i] if i in new else (prev["img"][old[i]], prev["mirror"][old[i]])
    np.savez(OUT / "rand_aug_dinov2_l.npz", ids=np.array(ids), img=np.array([get(i)[0] for i in ids], dtype=np.float32),
             mirror=np.array([get(i)[1] for i in ids], dtype=np.float32))
    log.info("saved features/rand_aug_dinov2_l.npz")


def cv():
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from aux_cv import ALPHA, BASE, MODELS, REPEATS, ROOT, ext, orig, proba, train, y

    t = np.load(ROOT / "features/overflow_tiles.npz")
    tpos = {i: k for k, i in enumerate(t["ids"])}
    TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
    TILE = ALPHA * np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1)
    if OPEN:   # open-bag score (bag_tiles.py), same x4 scale as aux_cv.py; views copy it like the tile scores
        b = np.load(ROOT / "features/bag_tiles.npz")
        bpos = {i: k for k, i in enumerate(b["ids"])}
        TILE = np.hstack([TILE, ALPHA * 4.0 * np.stack([b[k][[bpos[i] for i in train.id]]
                                                        for k in ("open_tile_max", "open_tile_top2")], 1)])
    X = np.hstack([BASE["dinov2 full"], TILE])
    r = np.load(OUT / "rand_aug_dinov2_l.npz")
    assert list(r["ids"]) == list(train.id)
    T = np.repeat(TILE[:, None], K, 1)
    avg = np.concatenate([unit(r["img"] + r["mirror"]), T], -1)                        # (N, K, D + 4)
    sep = np.concatenate([np.concatenate([r["img"], r["mirror"]], 1), np.repeat(T, 2, 1)], -1)   # (N, 2K, D + 4)

    variants = ({"no aug": (avg, 0), "+ 10 views": (avg, 10), "+ 30 views": (avg, 30)} if OPEN else
                {"no aug": (avg, 0), "+ 10 views": (avg, 10), "+ 30 views": (avg, 30), "+ 60 views": (avg, 60),
                 "+ 60 views, mirrors as own rows (120)": (sep, 2 * K)})
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
    watch = [26, 72, 19, 63, 13, 77, 41, 2, 60, 31, 101]
    wj = {w: int(np.where(train.id == w)[0][0]) for w in watch}
    rows, photos = [], {}
    for vname, (A, n) in variants.items():
        for mname, make in MODELS.items():
            t0 = time.time()
            P = {k: np.zeros((REPEATS, len(train), 4)) for k in ("no TTA", "TTA 60 views")}
            for s, (tr, va) in enumerate(splits):
                Xtr = np.vstack([X[tr], A[tr, :n].reshape(-1, A.shape[-1])])
                ytr = np.concatenate([y[tr], np.repeat(y[tr], n)])
                m = make().fit(Xtr, ytr)
                p0 = proba(m, X[va])
                pv = proba(m, avg[va].reshape(-1, avg.shape[-1])).reshape(len(va), K, -1).mean(1)
                P["no TTA"][s // 5, va] = p0
                P["TTA 60 views"][s // 5, va] = (p0 + K * pv) / (K + 1)
            for k, Pk in P.items():
                f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in Pk]
                rows.append({"train rows": vname, "TTA": k, "model": mname, "CV macro F1": np.mean(f1), "std": np.std(f1)})
                photos[f"{vname} | {k} | {mname}"] = {
                    f"#{w}": f"{Pk[:, j, y[j]].mean():.2f} ({(Pk[:, j].argmax(1) != y[j]).sum()}/{REPEATS})" for w, j in wj.items()}
                print(f"{vname:38s} | {k:12s} | {mname:20s} {np.mean(f1):.3f} ± {np.std(f1):.3f}", flush=True)
            log.info(f"cv {vname} | {mname} done in {time.time() - t0:.0f}s")
    res = pd.DataFrame(rows)
    tag = "_open" if OPEN else ""
    res.to_csv(ROOT / f"logs/rand_aug_cv{tag}.csv", index=False)
    pd.DataFrame(photos).to_csv(ROOT / f"logs/rand_aug_cv{tag}_photos.csv")
    pd.set_option("display.width", 250)
    print("\n" + res.pivot_table(index=["model", "TTA"], columns="train rows", values="CV macro F1")[list(variants)]
          .round(3).to_string())
    print("\nRBF SVM, per photo: mean p(true label) (times wrong out of 10)")
    print(pd.DataFrame(photos)[[f"{v} | {k} | RBF SVM" for v in variants for k in ("no TTA", "TTA 60 views")]].T.to_string())


OPEN = "--open" in sys.argv       # add the open-bag score (bag_tiles.py) to every row

if __name__ == "__main__":
    steps = [a for a in sys.argv[1:] if a != "--open"] or ["encode", "cv"]
    if "encode" in steps:
        encode()
    if "cv" in steps:
        cv()
