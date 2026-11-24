"""Add a batch of new external training photos and test whether they help.

The photos and their labels live in ../external_images/new_batch/ (see README.txt there):
batch.csv has one row per photo: file, label, source, license, url.
data/train.csv is never changed: the candidate training set is written to data/train_newbatch.csv,
and the CV scripts read it through the TRAIN_CSV environment variable.

  python add_batch.py check
      1. checks every row (file exists, label known, licence given; external data needs a stated licence)
      2. copies each photo to data/images/image_NNN_external.jpg with a new id (960 upward, never an id used before,
         since the feature files are keyed by id) and remembers the id in new_batch/assigned_ids.csv
      3. encodes the new photos with all 5 backbones (extract_features.py only encodes photos it has not seen)
      4. duplicate check: DINOv2 cosine of each new photo against all 102 competition photos, the current externals
         and the rest of the batch. Cosine >= DUP_COS is treated as the same photo (e.g. a web copy of a competition
         photo, as #911 was of #12) and left out. Report: logs/new_batch_dupes.csv, contact sheet logs/new_batch_sheet.jpg
      5. writes data/train_newbatch.csv = data/train.csv + the photos that passed, and prints the class balance

  python add_batch.py cv
      overflow whole-image + tile scores and open-bag scores for the new photos (overflow_aux.py score tiles, bag_tiles.py), then the
      same 10-repeat photo-grouped CV as aux_cv.py --open. External photos only ever sit in training folds and only the
      66 competition photos are scored, so the number is directly comparable with the current train.csv run (0.912
      for DINOv2 + tiles + open, RBF SVM). Results: logs/aux_cv_open_train_newbatch*.csv

Reference for the protocol: scikit-learn RepeatedStratifiedKFold; duplicate detection by embedding cosine similarity
(e.g. Oquab et al. 2023, DINOv2, which uses the same idea to remove near-duplicate images from its training data).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).parent
BATCH = ROOT.parent / "external_images" / "new_batch"
IMG_DIR = ROOT / "data" / "images"
OUT_CSV = ROOT / "data" / "train_newbatch.csv"
FIRST_ID, DUP_COS = 960, 0.90
SHORT = {"dishes_complete": "Washing Dishes|Complete", "dishes_incomplete": "Washing Dishes|Incomplete",
         "trash_complete": "Taking Out Trash|Complete", "trash_incomplete": "Taking Out Trash|Incomplete"}
PY = sys.executable


def read_batch():
    b = pd.read_csv(BATCH / "batch.csv", dtype=str).fillna("")
    b["label"] = b.label.str.strip().map(lambda s: SHORT.get(s.lower(), s))
    bad = []
    for r in b.itertuples():
        if not (BATCH / r.file).exists():
            bad.append(f"{r.file}: file not found in new_batch/")
        if r.label not in SHORT.values():
            bad.append(f"{r.file}: unknown label '{r.label}'")
        if not r.license.strip():
            bad.append(f"{r.file}: no licence given")
    if bad:
        sys.exit("batch.csv problems:\n  " + "\n  ".join(bad))
    if b.file.duplicated().any():
        sys.exit(f"batch.csv lists a file twice: {b.file[b.file.duplicated()].tolist()}")
    return b


def used_ids():
    """Every id already taken: image files on disk plus every id in any feature file (incl. removed photos)."""
    ids = {int(p.stem.split("_")[1]) for p in IMG_DIR.glob("image_*.jpg")}
    for f in (ROOT / "features").glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        if "ids" in z.files:
            ids |= {int(i) for i in z["ids"]}
    return ids


def assign_ids(b):
    path = BATCH / "assigned_ids.csv"
    known = dict(pd.read_csv(path).values) if path.exists() else {}
    taken, nxt = used_ids(), FIRST_ID
    for f in b.file:
        if f in known:
            continue
        while nxt in taken:
            nxt += 1
        known[f] = nxt
        taken.add(nxt)
    pd.DataFrame(sorted(known.items(), key=lambda kv: kv[1]), columns=["file", "id"]).to_csv(path, index=False)
    b["id"] = b.file.map(known).astype(int)
    for r in b.itertuples():
        dst = IMG_DIR / f"image_{r.id}_external.jpg"
        if not dst.exists():
            im = ImageOps.exif_transpose(Image.open(BATCH / r.file)).convert("RGB")
            im.save(dst, quality=95)
    return b


def full(name, ids):
    z = np.load(ROOT / f"features/{name}.npz")
    pos = {int(i): k for k, i in enumerate(z["ids"])}
    v = np.stack([z["img"][pos[i]] + z["img_flip"][pos[i]] for i in ids])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def thumb(path, h=220):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return im.resize((max(1, int(im.width * h / im.height)), h))


def check():
    b = read_batch()
    if b.empty:
        sys.exit("batch.csv has no rows yet: add photos to external_images/new_batch/ and one row each (see README.txt)")
    b = assign_ids(b)
    print(f"{len(b)} photos in batch, ids {b.id.min()}-{b.id.max()}; encoding new photos with 5 backbones ...", flush=True)
    subprocess.run([PY, "extract_features.py"], cwd=ROOT, check=True)

    train, test = pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/test.csv")
    ref = pd.concat([train.assign(group="train"), test.assign(group="test")])
    ref["group"] = np.where(ref.image_file.str.contains("_external"), "external", ref.group)
    paths = {int(p.stem.split("_")[1]): p for p in IMG_DIR.glob("image_*.jpg")}
    R, N = full("dinov2_l", ref.id.tolist()), full("dinov2_l", b.id.tolist())
    C, W = N @ R.T, N @ N.T
    np.fill_diagonal(W, -1)

    rows = []
    for k, r in enumerate(b.itertuples()):
        j = int(C[k].argmax())
        w = int(W[k].argmax())
        # keep the first of two near-identical batch photos with the SAME label; a same-scene pair with different
        # labels (e.g. own photo of a dirty bowl and the same bowl cleaned) is kept on purpose
        within = W[k, w] >= DUP_COS and k > w and b.label.iloc[k] == b.label.iloc[w]
        rows.append({"file": r.file, "id": r.id, "label": r.label, "nearest": f"#{ref.id.iloc[j]} ({ref.group.iloc[j]})",
                     "cos": round(float(C[k, j]), 3), "nearest_in_batch": b.file.iloc[w] if len(b) > 1 else "",
                     "cos_in_batch": round(float(W[k, w]), 3) if len(b) > 1 else np.nan,
                     "keep": C[k, j] < DUP_COS and not within, "ref_id": int(ref.id.iloc[j])})
    rep = pd.DataFrame(rows)
    rep.drop(columns="ref_id").to_csv(ROOT / "logs/new_batch_dupes.csv", index=False)

    # contact sheet: new photo | its nearest existing photo, red border when it would be dropped
    tiles = []
    for r in rep.itertuples():
        a, c = thumb(paths[r.id]), thumb(paths[r.ref_id])
        row = Image.new("RGB", (a.width + c.width + 30, 265), "white")
        row.paste(a, (0, 40)); row.paste(c, (a.width + 30, 40))
        d = ImageDraw.Draw(row)
        d.text((6, 4), f"NEW {r.file}  [{r.label}]", fill="black")
        d.text((6, 20), f"right: nearest {r.nearest}  cos {r.cos:.2f}" + ("" if r.keep else "  DROPPED"),
               fill="black" if r.keep else "red")
        if not r.keep:
            d.rectangle([0, 0, row.width - 1, row.height - 1], outline="red", width=4)
        tiles.append(row)
    if tiles:
        sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)), "white")
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y)); y += t.height
        sheet.save(ROOT / "logs/new_batch_sheet.jpg", quality=85)

    keep = b[rep.keep.values]
    new_rows = pd.DataFrame({"id": keep.id, "image_file": [f"image_{i}_external.jpg" for i in keep.id],
                             "image_url": keep.url, "label": keep.label})
    pd.concat([train, new_rows]).to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print("\n" + rep.drop(columns="ref_id").to_string(index=False))
    print(f"\nkept {len(keep)}/{len(b)} -> {OUT_CSV.relative_to(ROOT)}; contact sheet logs/new_batch_sheet.jpg")
    t = pd.read_csv(OUT_CSV)
    t["src"] = np.where(t.image_file.str.contains("_external"), "external", "competition")
    print("\nclass balance (competition + external):\n" + pd.crosstab(t.label, t.src, margins=True).to_string())


def cv():
    if not OUT_CSV.exists():
        sys.exit("run 'python add_batch.py check' first")
    env = {**os.environ, "TRAIN_CSV": str(OUT_CSV.relative_to(ROOT))}
    # overflow_aux.py score: whole-image dirty-bin score (aux_cv.py loads it for every training photo);
    # deterministic, so photos already scored keep the same values
    for cmd in (["overflow_aux.py", "score", "tiles"], ["bag_tiles.py"], ["aux_cv.py", "--open"]):
        print(f"\n===== {' '.join(cmd)} (TRAIN_CSV={env['TRAIN_CSV']}) =====", flush=True)
        subprocess.run([PY, *cmd], cwd=ROOT, env=env, check=True)
    base, new = ROOT / "logs/aux_cv_open.csv", ROOT / "logs/aux_cv_open_train_newbatch.csv"
    if base.exists():
        m = pd.read_csv(base).merge(pd.read_csv(new), on=["variant", "model"], suffixes=(" train.csv", " new batch"))
        m["change"] = m["CV macro F1 new batch"] - m["CV macro F1 train.csv"]
        print("\n" + m.round(3).to_string(index=False))
        print("\nDifferences under ~0.02 are within the noise of 10 repeats.")


if __name__ == "__main__":
    {"check": check, "cv": cv}.get((sys.argv[1:] or [""])[0], lambda: sys.exit(__doc__))()
