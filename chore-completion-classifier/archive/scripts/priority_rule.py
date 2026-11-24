"""Priority rule: "soiled dishes anywhere beat a clean sink" on top of the 4-class model.

Idea (user's): a photo can show a clean sink AND a soiled dish (e.g. #2: clean sink, small plate with food on the
counter). The whole-image model weighs both and often says Complete. Rule: if the photo is a dishes photo and the
soiled-dish detector finds a soiled patch, the answer is Washing Dishes|Incomplete, whatever the model said.

Pieces:
  base model     : DINOv2 + tiles + open-bag score, RBF SVM (aux_cv.py --open), trained on TRAIN_CSV
                   (default data/train_newbatch_clean.csv = train.csv + 7 licence-clean dishes photos)
  soiled detector: soiled_windows.py's window classifier (74 DINOv2 windows per photo, window labels from the
                   hand-drawn soiled boxes in dish_boxes.py, plus the hand crops), trained only on training photos;
                   soiled_max = highest window score of a photo
  rule           : if the base model's chore is dishes and soiled_max > t -> Washing Dishes|Incomplete

The threshold t is never chosen on the photo being scored. Inside each outer training fold, an inner 5-fold CV gives
held-out base predictions and soiled_max for every training photo, and t is picked from those:
  t best  : the t that gives the best macro F1 on the training photos (ties -> the highest t, i.e. fewest overrides)
  t safe  : the highest soiled_max of any Complete dishes training photo ("more soiled than every clean photo seen")
Outer CV: the same 10 x 5 photo-grouped splits as aux_cv.py; only the 66 competition photos are scored.
-> logs/priority_rule.csv (F1 per variant), logs/priority_rule_photos.csv (wrong counts), printed summary
With --plates: the soiled detector also trains on the 40 labelled plates of the Kaggle "Clean or Dirty plate dataset"
(ssalibenjamin/clean-or-dirty-plate-dataset, MIT per the uploader; originally the platesv2 competition data):
train/dirty = soiled (1), train/cleaned = clean (0). Each plate and its mirror are DINOv2-encoded
-> features/plates_dinov2_l.npz; outputs get a _plates suffix.
Usage: python priority_rule.py [--plates]
"""
import os
import sys

os.environ.setdefault("TRAIN_CSV", "data/train_newbatch_clean.csv")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import aux_cv as A
from extract_features import OUT
from soiled_windows import SKIP, window_labels

PLATES = "--plates" in sys.argv
TAG = "_plates" if PLATES else ""
PLATE_DIR = A.ROOT.parent / "external_data/clean_dirty_plates/plates/train"

y, train, orig, ext = A.y, A.train, A.orig, A.ext
ids = train.id.values

# base features: DINOv2 + overflow tiles + open-bag score (same as aux_cv.py --open)
t = np.load(OUT / "overflow_tiles.npz"); tp = {i: k for k, i in enumerate(t["ids"])}
b = np.load(OUT / "bag_tiles.npz"); bp = {i: k for k, i in enumerate(b["ids"])}
TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
X = np.hstack([A.BASE["dinov2 full"], A.ALPHA * np.stack([t[k][[tp[i] for i in ids]] for k in TK], 1),
               A.ALPHA * 4.0 * np.stack([b[k][[bp[i] for i in ids]] for k in ("open_tile_max", "open_tile_top2")], 1)])

# soiled-dish windows (competition dishes photos only) and hand crops
z = np.load(OUT / "soiled_windows.npz"); E = z["emb"]; wpos = {int(i): k for k, i in enumerate(z["ids"])}
cr = np.load(OUT / "dish_crops.npz"); cpos = {int(i): k for k, i in enumerate(cr["ids"])}
DISH = {int(ids[j]) for j in orig if y[j] < 2}
LAB = {i: window_labels(i, y[np.where(ids == i)[0][0]] == 1) for i in DISH}


def plate_features():
    """DINOv2 (CLS + mean patch, same preprocessing as extract_features.py) of the 40 labelled plates + mirrors."""
    path = OUT / "plates_dinov2_l.npz"
    if path.exists():
        z = np.load(path)
        return z["emb"], z["y"]
    import torch
    from PIL import Image, ImageOps
    from extract_features import load_model, unit
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc, pre = load_model("dinov2_l", device)
    emb, lab, names = [], [], []
    with torch.no_grad():
        for cls, v in (("cleaned", 0), ("dirty", 1)):
            for f in sorted((PLATE_DIR / cls).glob("*.jpg")):
                im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                e = unit(enc(torch.stack([pre(im), pre(ImageOps.mirror(im))]).to(device)))
                emb += list(e); lab += [v, v]; names += [f"{cls}/{f.name}"] * 2
    np.savez(path, emb=np.array(emb, dtype=np.float32), y=np.array(lab), names=np.array(names))
    return np.array(emb, dtype=np.float32), np.array(lab)


PE, PY = plate_features() if PLATES else (None, None)


def fit_detector(photo_ids):
    Xs, ys = [], []
    if PLATES:
        Xs.append(PE); ys.append(PY)
    for i in photo_ids:
        m = LAB[i] >= 0
        Xs.append(E[wpos[i]][m]); ys.append(LAB[i][m])
        if i in cpos and i not in SKIP:
            Xs.append(cr["feats"][cpos[i]]); ys.append(np.full(cr["feats"].shape[1], cr["y"][cpos[i]]))
    return LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(np.vstack(Xs), np.concatenate(ys))


def soiled_max(det, rows):
    """soiled_max for each row; NaN for photos without windows (trash and external photos)."""
    return np.array([det.predict_proba(E[wpos[int(ids[j])]])[:, 1].max() if int(ids[j]) in wpos else np.nan
                     for j in rows])


def base_pred(rows_tr, rows_va):
    return A.MODELS["RBF SVM"]().fit(X[rows_tr], y[rows_tr]).predict(X[rows_va])


def apply_rule(pred, smax, thr):
    out = pred.copy()
    out[(pred < 2) & ~np.isnan(smax) & (smax > thr)] = 1          # dishes chore + soiled patch -> Dishes Incomplete
    return out


def macro(yt, yp):
    return f1_score(yt, yp, average="macro")


def main():
    splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
              RepeatedStratifiedKFold(n_splits=5, n_repeats=A.REPEATS, random_state=0).split(orig, y[orig])]
    V = ["base (no rule)", "rule, t best", "rule, t safe"]
    P = {v: np.zeros((A.REPEATS, len(y)), int) for v in V}
    SM = np.full((A.REPEATS, len(y)), np.nan)          # held-out soiled_max of each photo
    T = {v: [] for v in V[1:]}
    for k, (tr, va) in enumerate(splits):
        tro = tr[np.isin(tr, orig)]
        # inner CV on the training photos -> held-out base predictions and soiled_max for each of them
        ip, ism = np.zeros(len(tro), int), np.full(len(tro), np.nan)
        for itr, iva in StratifiedKFold(5, shuffle=True, random_state=k).split(tro, y[tro]):
            ip[iva] = base_pred(np.r_[tro[itr], ext], tro[iva])
            ism[iva] = soiled_max(fit_detector([int(ids[j]) for j in tro[itr] if int(ids[j]) in DISH]), tro[iva])
        cand = np.r_[np.unique(ism[~np.isnan(ism)]), 1.0]
        scores = [macro(y[tro], apply_rule(ip, ism, c)) for c in cand]
        t_best = cand[max(range(len(cand)), key=lambda q: (round(scores[q], 6), cand[q]))]
        comp = ~np.isnan(ism) & (y[tro] == 0)
        t_safe = ism[comp].max() if comp.any() else 1.0
        # outer fold: train on all training rows, score the held-out competition photos
        pred = base_pred(tr, va)
        sm = soiled_max(fit_detector([int(ids[j]) for j in tro if int(ids[j]) in DISH]), va)
        r = k // 5
        SM[r, va] = sm
        P[V[0]][r, va] = pred
        P[V[1]][r, va] = apply_rule(pred, sm, t_best); T[V[1]].append(t_best)
        P[V[2]][r, va] = apply_rule(pred, sm, t_safe); T[V[2]].append(t_safe)
        print(f"fold {k + 1}/{len(splits)}  t best {t_best:.2f}  t safe {t_safe:.2f}", flush=True)

    dsh, trs = orig[y[orig] < 2], orig[y[orig] >= 2]
    rows, wrong = [], {}
    for v in V:
        f = [macro(y[orig], p[orig]) for p in P[v]]
        fd = [f1_score(y[dsh], p[dsh], labels=[0, 1], average="macro") for p in P[v]]
        ft = [f1_score(y[trs], p[trs], labels=[2, 3], average="macro") for p in P[v]]
        rows.append({"variant": v, "CV macro F1": np.mean(f), "std": np.std(f), "dishes F1": np.mean(fd),
                     "trash F1": np.mean(ft), "mean t": np.mean(T[v]) if v in T else np.nan})
        wrong[v] = {f"#{ids[j]}": int((P[v][:, j] != y[j]).sum()) for j in dsh}
    res = pd.DataFrame(rows); res.to_csv(A.ROOT / f"logs/priority_rule{TAG}.csv", index=False)
    w = pd.DataFrame(wrong); w = w[(w > 0).any(axis=1)]
    w.to_csv(A.ROOT / f"logs/priority_rule{TAG}_photos.csv")
    pd.set_option("display.width", 200)
    print("\n" + res.round(3).to_string(index=False))
    print("\nDishes photos wrong at least once (times wrong out of 10):\n" + w.to_string())
    from sklearn.metrics import roc_auc_score
    auc = [roc_auc_score(y[dsh], SM[r, dsh]) for r in range(A.REPEATS)]
    print(f"\nsoiled_max alone, Incomplete vs Complete on the 33 dishes photos: AUC {np.mean(auc):.3f} ± {np.std(auc):.3f}")
    order = sorted(dsh, key=lambda j: np.nanmean(SM[:, j]))
    print("held-out soiled_max (C = Complete, I = Incomplete): " +
          ", ".join(f"#{ids[j]}{'I' if y[j] == 1 else 'C'} {np.nanmean(SM[:, j]):.2f}" for j in order))


if __name__ == "__main__":
    main()
