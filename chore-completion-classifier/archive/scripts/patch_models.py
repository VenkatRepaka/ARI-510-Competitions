"""Patch-level models on DINOv2 patch tokens (features/dinov2_l_patches.npz, from extract_patches.py).

Two steps: (1) chore type from logistic regression on the CLS token (always right so far),
(2) complete vs incomplete *within that chore* from patch-level evidence:
  pool    logistic regression / RBF SVM on [CLS, mean patch, max patch] vectors
  patchLR logistic regression on single patches (each patch gets its photo's label);
          photo score = mean logit of its top-k% most "incomplete" patches
  nbnn    naive-Bayes nearest neighbour: each patch's best match among incomplete vs complete training
          photos of the same chore; photo score = mean (or top/bottom-k) of the differences
Each variant is scored alone and averaged with the baseline (all backbones | full | extra trees).
Same 10x repeated 5-fold CV over the 66 original photos as train.py; web images are not used.
Writes logs/patch_results.csv and features/patch_oof.npz (out-of-fold probabilities, for later use).

Usage: python patch_models.py
"""
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

import train as T
from extract_features import CLASSES, LOG_DIR, OUT, unit

BACKBONES = ["dinov2_l", "eva02_l", "siglip_so400m", "vith14_laion", "vitl14_openai"]
rng = np.random.default_rng(0)

z = np.load(OUT / "dinov2_l_patches.npz")
pos = {i: k for k, i in enumerate(z["ids"])}
orig = T.orig_idx
ids = T.train.id.values[orig]
y = T.y[orig]
chore = y // 2          # 0 dishes, 1 trash
incomplete = y % 2      # 0 complete, 1 incomplete
P = z["patches"][[pos[i] for i in ids]].astype(np.float32)   # (66, 1024, 1024)
CLS = z["cls"][[pos[i] for i in ids]]
n, n_patch, dim = P.shape
POOL = unit(np.hstack([CLS, unit(P.mean(1)), unit(P.max(1))]))


def best_match_sims():
    """S[i, j, p] = highest cosine similarity between patch p of photo i and any patch of photo j."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Pt = torch.tensor(P, device=dev)
    S = np.zeros((n, n, n_patch), np.float32)
    for i in range(n):
        S[i] = torch.einsum("pd,jqd->jpq", Pt[i], Pt).amax(2).cpu().numpy()
    S[np.arange(n), np.arange(n)] = np.nan   # a photo never matches itself
    return S


def nbnn_scores(S, photos, train_idx, agg):
    """Score > 0 = looks incomplete. Compares against training photos of the photo's own chore."""
    out = []
    for i in photos:
        others = [j for j in train_idx if j != i and chore[j] == chore[i]]
        inc = [j for j in others if incomplete[j]]
        com = [j for j in others if not incomplete[j]]
        m = min(len(inc), len(com))   # equal photo counts, so the bigger class does not win by default
        draw = np.random.default_rng(i)
        d = np.mean([np.nanmax(S[i, draw.choice(inc, m, replace=False)], 0)
                     - np.nanmax(S[i, draw.choice(com, m, replace=False)], 0) for _ in range(5)], 0)
        if agg == "mean":
            out.append(d.mean())
        else:
            k = int(len(d) * agg)
            d = np.sort(d)
            out.append(d[-k:].mean() + d[:k].mean())
    return np.array(out)


def patch_lr_scores(train_idx, test_idx, topks, per_photo=128):
    """Train a patch classifier per chore on random patches, score photos by their top-k% patch logits."""
    out = {k: np.zeros(len(test_idx)) for k in topks}
    for c in (0, 1):
        tr = [i for i in train_idx if chore[i] == c]
        pick = [rng.choice(n_patch, per_photo, replace=False) for _ in tr]
        X = np.vstack([P[i, p] for i, p in zip(tr, pick)])
        yy = np.repeat(incomplete[tr], per_photo)
        clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X, yy)
        for k, i in enumerate(test_idx):
            if chore_pred_cache[i] != c:
                continue
            logit = np.sort(clf.decision_function(P[i]))
            for t in topks:
                out[t][k] = logit[-max(1, int(n_patch * t)):].mean()
    return out


def to_4class(chore_pred, p_inc):
    return np.stack([(chore_pred == 0) * (1 - p_inc), (chore_pred == 0) * p_inc,
                     (chore_pred == 1) * (1 - p_inc), (chore_pred == 1) * p_inc], 1)


def platt(train_scores, train_y, test_scores):
    lr = LogisticRegression(C=1.0).fit(train_scores[:, None], train_y)
    return lr.predict_proba(test_scores[:, None])[:, 1]


if __name__ == "__main__":
    t0 = time.time()
    log = T.get_logger("patch", f"patch_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log.info(f"{n} photos, {n_patch} patches of dim {dim}. Computing patch best-match table...")
    S = best_match_sims()
    log.info(f"done in {time.time() - t0:.0f}s")
    F = {b: T.load(b) for b in BACKBONES}
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(y, y))
    variants = ["baseline extra trees", "pool logreg", "pool RBF SVM", "patchLR top5%", "patchLR top10%",
                "patchLR top25%", "patchLR mean", "nbnn mean", "nbnn top/bottom10%", "nbnn top/bottom25%"]
    oof = {v: [] for v in variants}
    for f, (tr, va) in enumerate(splits):
        base = np.mean([T.fit_predict(F, b, "full", "extra trees", orig[tr], "tr", orig[va]) for b in BACKBONES], 0)
        oof["baseline extra trees"].append(base)
        chore_pred = LogisticRegression(C=1, max_iter=3000).fit(CLS[tr], chore[tr]).predict(CLS[va])
        chore_pred_cache = dict(zip(va, chore_pred))
        for name, make in (("pool logreg", lambda: LogisticRegression(C=10, max_iter=5000, class_weight="balanced")),
                           ("pool RBF SVM", lambda: SVC(C=3, probability=True, class_weight="balanced", random_state=0))):
            p_inc = np.zeros(len(va))
            for c in (0, 1):
                m_tr, m_va = chore[tr] == c, chore_pred == c
                if m_va.any():
                    p_inc[m_va] = make().fit(POOL[tr][m_tr], incomplete[tr][m_tr]).predict_proba(POOL[va][m_va])[:, 1]
            oof[name].append(to_4class(chore_pred, p_inc))
        TOPKS = (.05, .10, .25, 1.0)
        lr_scores = patch_lr_scores(tr, va, TOPKS)
        chore_pred_cache.update({i: chore[i] for i in tr})   # inner folds use the true chore
        inner = {k: np.zeros(len(tr)) for k in TOPKS}
        for itr, iva in RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=f).split(tr, y[tr]):
            sc = patch_lr_scores(tr[itr], tr[iva], TOPKS)
            for k in TOPKS:
                inner[k][iva] = sc[k]
        for name, k in (("patchLR top5%", .05), ("patchLR top10%", .10), ("patchLR top25%", .25), ("patchLR mean", 1.0)):
            p_inc = np.zeros(len(va))
            for c in (0, 1):   # calibrate per chore
                mt, mv = chore[tr] == c, chore_pred == c
                if mv.any():
                    p_inc[mv] = platt(inner[k][mt], incomplete[tr][mt], lr_scores[k][mv])
            oof[name].append(to_4class(chore_pred, p_inc))
        for name, agg in (("nbnn mean", "mean"), ("nbnn top/bottom10%", .10), ("nbnn top/bottom25%", .25)):
            s_tr, s_va = nbnn_scores(S, tr, tr, agg), nbnn_scores(S, va, tr, agg)
            p_inc = np.zeros(len(va))
            for c in (0, 1):   # calibrate per chore
                mt, mv = chore[tr] == c, chore_pred == c
                if mv.any():
                    p_inc[mv] = platt(s_tr[mt], incomplete[tr][mt], s_va[mv])
            oof[name].append(to_4class(chore_pred, p_inc))
        if (f + 1) % 10 == 0:
            log.info(f"fold {f + 1}/{len(splits)}  {time.time() - t0:.0f}s")

    def score(probs):
        s = [f1_score(y[va], p.argmax(1), average="macro") for (_, va), p in zip(splits, probs)]
        return np.mean(s), np.std(s)

    rows = []
    for v in variants:
        rows.append((v, *score(oof[v])))
        if v != "baseline extra trees":
            rows.append((f"{v} + baseline", *score([(a + b) / 2 for a, b in zip(oof[v], oof["baseline extra trees"])])))
    res = pd.DataFrame(rows, columns=["entry", "cv_macro_f1", "std"]).sort_values("cv_macro_f1", ascending=False)
    res.to_csv(LOG_DIR / "patch_results.csv", index=False)
    log.info("\n" + res.round(3).to_string(index=False))
    np.savez(OUT / "patch_oof.npz", **{v: np.array(oof[v], dtype=object) for v in variants},
             splits=np.array([va for _, va in splits], dtype=object))
    # per-photo error rate of the best patch variant, for the photos we keep missing
    best = res[~res.entry.str.startswith("baseline")].iloc[0].entry
    probs = [(a + b) / 2 for a, b in zip(oof[best.replace(" + baseline", "")], oof["baseline extra trees"])] \
        if best.endswith("+ baseline") else oof[best]
    for label, pr in (("baseline", oof["baseline extra trees"]), (best, probs)):
        err = np.zeros(n)
        for (_, va), p in zip(splits, pr):
            err[va] += p.argmax(1) != y[va]
        hard = {i: e / 10 for i, e in zip(ids, err) if e >= 5}
        log.info(f"{label}: photos wrong in >=50% of repeats: {hard}")
    log.info(f"done in {(time.time() - t0) / 60:.1f} min")
