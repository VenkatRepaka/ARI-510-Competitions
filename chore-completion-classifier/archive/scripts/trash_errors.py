"""Misclassified trash photos for the current candidate (DINOv2 full + overflow tiles + open-bag score, RBF SVM),
10 repeats of photo-grouped 5-fold CV, same splits as aux_cv.py. Writes logs/trash_errors.csv and logs/trash_errors.jpg."""
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import RepeatedStratifiedKFold
import aux_cv as A

t = np.load(A.ROOT / "features/overflow_tiles.npz"); tp = {i: k for k, i in enumerate(t["ids"])}
TILE = np.stack([t[k][[tp[i] for i in A.train.id]] for k in
                 ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]], 1)
b = np.load(A.ROOT / "features/bag_tiles.npz"); bp = {i: k for k, i in enumerate(b["ids"])}
OPEN = 4.0 * np.stack([b[k][[bp[i] for i in A.train.id]] for k in ("open_tile_max", "open_tile_top2")], 1)
X = np.hstack([A.BASE["dinov2 full"], TILE, OPEN])
y, orig, ext = A.y, A.orig, A.ext
splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
          RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig])]
P = np.zeros((A.REPEATS, len(y), 4))
for k, (tr, va) in enumerate(splits):
    P[k // 5, va] = A.proba(A.MODELS["RBF SVM"]().fit(X[tr], y[tr]), X[va])
rows = []
for j in orig:
    if y[j] not in (2, 3):
        continue
    pred = P[:, j].argmax(1)
    wrong = int((pred != y[j]).sum())
    if wrong:
        rows.append({"id": int(A.train.id[j]), "label": A.CLASSES[y[j]], "wrong": f"{wrong}/{A.REPEATS}",
                     "usually predicted": A.CLASSES[np.bincount(pred, minlength=4).argmax()], "p_true": round(P[:, j, y[j]].mean(), 2)})
df = pd.DataFrame(rows).sort_values("p_true")
df.to_csv(A.ROOT / "logs/trash_errors.csv", index=False)
print(df.to_string(index=False))
H, ims = 330, []
for r in df.itertuples():
    im = ImageOps.exif_transpose(Image.open(A.ROOT / f"data/images/image_{r.id:03d}.jpg")).convert("RGB")
    ims.append((r, im.resize((int(im.width * H / im.height), H))))
W = sum(m.width for _, m in ims) + 12 * len(ims)
c = Image.new("RGB", (W, H + 48), "white"); d = ImageDraw.Draw(c); x = 0
for r, im in ims:
    c.paste(im, (x, 0))
    d.text((x + 3, H + 4), f"#{r.id}  true: {r.label.split('|')[1]}", fill="black")
    d.text((x + 3, H + 18), f"pred: {r._4.split('|')[1]}  wrong {r.wrong}", fill="red")
    d.text((x + 3, H + 32), f"p(true) {r.p_true}", fill="gray")
    x += im.width + 12
c.save(A.ROOT / "logs/trash_errors.jpg")
