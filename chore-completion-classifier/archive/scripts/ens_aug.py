"""Ensemble: submitted RBF SVM (no augmentation) + logistic regression trained with +10 random views per photo
(features/rand_aug_dinov2_l.npz from rand_aug.py). Probabilities are averaged with weight w on the SVM.
Same 10x5 photo-grouped CV as aux_cv.py; the dev fold is scored on original photos only.
w = 0.5 is the pre-chosen main result; the other weights are shown for reference (picking the best is optimistic).
  -> logs/ens_aug_cv.csv, logs/ens_aug_cv_photos.csv
Usage: python ens_aug.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from aux_cv import ALPHA, BASE, MODELS, REPEATS, ROOT, ext, orig, proba, train, y
from extract_features import CLASSES, OUT, unit

N_VIEWS = 10
t = np.load(ROOT / "features/overflow_tiles.npz")
tpos = {i: k for k, i in enumerate(t["ids"])}
TK = ["dinov2_l_tile_max", "dinov2_l_tile_top2", "siglip_so400m_tile_max", "siglip_so400m_tile_top2"]
TILE = ALPHA * np.stack([t[k][[tpos[i] for i in train.id]] for k in TK], 1)
X = np.hstack([BASE["dinov2 full"], TILE])
r = np.load(OUT / "rand_aug_dinov2_l.npz")
assert list(r["ids"]) == list(train.id)
A = np.concatenate([unit(r["img"] + r["mirror"]), np.repeat(TILE[:, None], r["img"].shape[1], 1)], -1)

splits = [(np.r_[orig[tr], ext], orig[va]) for tr, va in
          RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(orig, y[orig])]
svm, lr = (np.zeros((REPEATS, len(train), 4)) for _ in range(2))
for s, (tr, va) in enumerate(splits):
    svm[s // 5, va] = proba(MODELS["RBF SVM"]().fit(X[tr], y[tr]), X[va])
    Xtr = np.vstack([X[tr], A[tr, :N_VIEWS].reshape(-1, A.shape[-1])])
    ytr = np.concatenate([y[tr], np.repeat(y[tr], N_VIEWS)])
    lr[s // 5, va] = proba(MODELS["logistic regression"]().fit(Xtr, ytr), X[va])

watch = [26, 72, 19, 63, 13, 77, 41, 2, 60, 31, 101]
wj = {w: int(np.where(train.id == w)[0][0]) for w in watch}
rows, photos = [], {}
for w in (1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0):
    P = w * svm + (1 - w) * lr
    f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
    name = {1.0: "SVM only", 0.0: "LR +10 views only"}.get(w, f"{w:.1f} SVM + {1 - w:.1f} LR")
    rows.append({"ensemble": name, "SVM weight": w, "CV macro F1": np.mean(f1), "std": np.std(f1)})
    photos[name] = {f"#{k}": f"{P[:, j, y[j]].mean():.2f} ({(P[:, j].argmax(1) != y[j]).sum()}/{REPEATS})" for k, j in wj.items()}
res = pd.DataFrame(rows)
res.to_csv(ROOT / "logs/ens_aug_cv.csv", index=False)
pd.DataFrame(photos).to_csv(ROOT / "logs/ens_aug_cv_photos.csv")
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
print("\nper photo: mean p(true label) (times wrong out of 10)\n" + pd.DataFrame(photos).to_string())
# how often the two models disagree on the dev photos (argmax), averaged over repeats
print(f"\nSVM vs LR disagree on {np.mean([(a[orig].argmax(1) != b[orig].argmax(1)).mean() for a, b in zip(svm, lr)]):.1%} of dev photos")

# dishes only: macro F1 over the two dishes classes, on the original dishes dev photos, and errors per repeat
dish = orig[np.isin(train.label.iloc[orig].str.startswith("Washing Dishes").values, True)]
print("\nDishes only (original dishes photos):")
for w in (1.0, 0.5, 0.0):
    P = w * svm + (1 - w) * lr
    pred = P[:, dish].argmax(2)
    f1 = [f1_score(y[dish], p, labels=np.unique(y[dish]), average="macro") for p in pred]
    wrong = (pred != y[dish]).sum(1).mean()
    print(f"  SVM weight {w:.1f}: dishes F1 {np.mean(f1):.3f} ± {np.std(f1):.3f}, wrong {wrong:.1f} of {len(dish)} per repeat")

# routed by chore: the chore (dishes vs trash) comes from the SVM (sum of its two class probabilities); within the chore
# Complete vs Incomplete comes from the SVM for trash and from the +10-views logistic regression for dishes.
D = [k for k, c in enumerate(CLASSES) if c.startswith("Washing Dishes")]
TR = [k for k, c in enumerate(CLASSES) if c.startswith("Taking Out Trash")]
route = svm.copy()
is_dish = svm[..., D].sum(-1) > svm[..., TR].sum(-1)
for k in D:
    route[..., k] = np.where(is_dish, lr[..., k] / lr[..., D].sum(-1), 0)
for k in TR:
    route[..., k] = np.where(is_dish, 0, svm[..., k] / svm[..., TR].sum(-1))
print("\nRouted (chore from SVM; trash: SVM, dishes: LR +10 views):")
for name, P in (("SVM only", svm), ("routed", route)):
    f1 = [f1_score(y[orig], p[orig].argmax(1), average="macro") for p in P]
    fd = [f1_score(y[dish], p[dish].argmax(1), labels=np.unique(y[dish]), average="macro") for p in P]
    print(f"  {name:9s} CV macro F1 {np.mean(f1):.3f} ± {np.std(f1):.3f}   dishes F1 {np.mean(fd):.3f}   "
          f"chore wrong {np.mean([(p[orig][:, D].sum(1) > 0.5) != np.isin(y[orig], D) for p in P]) * len(orig):.2f}/{len(orig)}")
wrong = {f"#{train.id.iloc[j]}": f"{(svm[:, j].argmax(1) != y[j]).sum()} -> {(route[:, j].argmax(1) != y[j]).sum()}"
         for j in orig if (svm[:, j].argmax(1) != y[j]).any() or (route[:, j].argmax(1) != y[j]).any()}
print("  photos ever wrong (times wrong of 10, SVM -> routed):", wrong)
