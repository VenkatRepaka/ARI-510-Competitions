"""Train classifiers on the saved features and score them (the experiment loop of this project).

WHAT THIS FILE DOES
    The feature scripts turned every image into numbers (quality_features.py: 74 per image; embed_features.py:
    768-2048 per network). This file takes one or more of those feature sets, trains several classifiers on
    them, and reports how well each one predicts artifact_present. Every number in the results tables of the
    README comes from here.

    It also provides small helper functions that other scripts import:
        load_labels, load_feature, matrix, models   -> used by make_submission.py and review_mistakes.py

THE METRIC: binary F1 for the class "artifact present" (1)
    precision = of the images predicted "artifact", the share that really have one
    recall    = of the images that really have an artifact, the share that were found
    F1        = 2 * precision * recall / (precision + recall)   = 2TP / (2TP + FP + FN)   [1]
    This is the competition's metric. A classifier outputs a probability; it becomes 0/1 with a threshold
    (0.5 unless stated otherwise).
    Floor: predicting 1 for every validation image gives F1 0.703 (64 of 118 validation images are 1:
    2*64 / (2*64 + 54) = 0.703), so a useful model must beat that.

TWO SCORES FOR EVERY MODEL
    dev F1  Fit on train.csv (823 images), predict validation.csv (118 images): the split the host intends for
            development. With only 118 images one image changes F1 by about 0.01, so this number is noisy.
    CV F1   Cross-validation [2] on train + validation together (941 images):
              - split the 941 images into 5 parts ("folds") with the same share of 1s in each ("stratified");
              - train on 4 parts, predict the 5th; rotate so every part is predicted once;
              - repeat the whole thing 3 times with different random splits.
            Every image is then predicted 3 times, each time by a model that never saw it. These are the
            OUT-OF-FOLD (OOF) probabilities. F1 is computed per repeat; the mean +- standard deviation over the
            3 repeats is reported. Uses 8x more images for scoring than dev, so it is the number used to choose
            models.
            Caveat: the near-duplicate image pairs (logs/near_duplicates.csv) can land in different folds, which
            makes CV very slightly optimistic.
    tuned   F1 at the best single threshold (0.10-0.90) on the OOF probabilities averaged over the 3 repeats.
            Optimistic, because the threshold is chosen on the same data it is scored on [3]. Only used to check
            that 0.5 is close to the best threshold (it was: 0.41-0.48 for the good models).

THE CLASSIFIERS (all from scikit-learn [4], except the MLP from mlp.py)
    logreg      Logistic regression [5]: a weighted sum of the features -> probability. C = 0.1 is fairly strong
                regularisation (keeps the weights small), sensible with many features and few images.
    svm_rbf     Support vector machine [6] with an RBF (Gaussian) kernel: can draw curved boundaries between the
                classes. C = 3 sets how hard it tries to classify every training image correctly.
                probability=True adds Platt scaling [7] (a small internal cross-validation) so it outputs
                probabilities. Best model on every feature set here.
    hgb         Histogram gradient boosting: many small decision trees, each correcting the previous ones
                [8][9]. Trees do not need scaled features. Slow on thousands of features.
    mlp, mlp_h64, mlp_strong   the PyTorch neural network in mlp.py (256 hidden units; 64 hidden units;
                stronger regularisation: input dropout 0.5, weight decay 0.1).
    StandardScaler (logreg, svm): rescales each feature to mean 0 and standard deviation 1. SVMs and logistic
    regression are sensitive to feature scale [10]; without it the features with big numbers would dominate.
    The scaler is part of the pipeline, so in CV it is fitted on the training folds only (no information from
    the held-out fold leaks in).
    The settings (C = 0.1, C = 3, 300 trees, learning rate 0.05) are common starting values and were not tuned.

OUTPUTS
    logs/<out_csv>                      one row per (feature set, model): dev_f1, cv_f1, cv_sd, cv_tuned, thr
    features/oof_<set>_<model>.npy      OOF probabilities (941 values, mean of the 3 repeats), in the order
                                        train.csv rows then validation.csv rows. Used by review_mistakes.py.

USAGE
    python baseline.py
        Standard grid: each feature file alone, each network + quality, and all three networks + quality, with
        every model in models() (now 6: the 3 MLPs were added later, so a full run takes much longer than the
        first one did). Results -> logs/baseline.csv.
    python baseline.py <out.csv> <model,model,...> <set> [<set> ...]
        Only the named models and feature sets; a set joins feature files with "+". Example:
        python baseline.py mlp.csv mlp,mlp_strong eva02_l+quality siglip_so400m+quality

REPRODUCIBILITY
    All random choices use SEED = 0 (CV splits, SVM's Platt scaling, boosting, MLP), so reruns give the same
    numbers. The test set is never used here.

REFERENCES
    [1] van Rijsbergen, C. J. (1979). Information Retrieval, 2nd ed. Butterworths. (The F-measure.)
    [2] Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model
        selection. Proc. IJCAI 1995. (Recommends stratified cross-validation.)
    [3] Lipton, Z. C., Elkan, C., Naryanaswamy, B. (2014). Optimal thresholding of classifiers to maximize F1
        measure. Proc. ECML PKDD 2014.
    [4] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning
        Research 12, 2825-2830.
    [5] Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical
        Society B 20(2), 215-242.
    [6] Cortes, C., Vapnik, V. (1995). Support-vector networks. Machine Learning 20(3), 273-297.
    [7] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
        likelihood methods. In Advances in Large Margin Classifiers, MIT Press.
    [8] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of
        Statistics 29(5), 1189-1232.
    [9] Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Proc. NeurIPS 2017.
        (The histogram method that scikit-learn's HistGradientBoostingClassifier follows.)
    [10] Hsu, C.-W., Chang, C.-C., Lin, C.-J. (2003). A practical guide to support vector classification.
        Technical report, National Taiwan University. (Scaling features before an SVM; the RBF kernel.)
"""
from pathlib import Path   # building file paths

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier   # gradient-boosted trees
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold    # the repeated, stratified CV splitter
from sklearn.pipeline import make_pipeline                     # chain "scale, then classify" into one model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC                                    # support vector classifier

from mlp import MLPClassifier   # the PyTorch neural network (mlp.py), with the same fit / predict_proba interface

ROOT = Path(__file__).parent    # the folder this script is in
FEAT = ROOT / "features"        # the saved feature files (*.npz) and OOF predictions (oof_*.npy)
SEED = 0                        # one seed for every random choice, so results are reproducible


# ----------------------------------------------------------------------------------------------------------
# Loading data
# ----------------------------------------------------------------------------------------------------------
def load_labels():
    """The two labelled tables: train (823 rows) and validation (118 rows).
    Columns: image_id, filename, artifact_present (0 or 1)."""
    tr = pd.read_csv(ROOT / "data/train.csv")
    va = pd.read_csv(ROOT / "data/validation.csv")
    return tr, va


def load_feature(name):
    """Load features/<name>.npz and return a dictionary {image_id: vector}.

    Quality-type files (quality, bands, tilesim_..., tileagg_...) store their numbers under "X" -> used as they are.
    Embedding files (dinov2_l, siglip_so400m, eva02_l) store "img" and "img_flip" -> their average is used
    (the image and its mirror image describe the same photo; averaging makes the description more stable).
    A dictionary keyed by image_id means rows are looked up by id, never by position, so feature files can
    never be matched to the wrong image.
    """
    z = np.load(FEAT / f"{name}.npz", allow_pickle=True)   # allow_pickle: ids are stored as Python strings
    X = z["X"] if "X" in z else (z["img"] + z["img_flip"]) / 2
    return dict(zip(z["ids"], X))


def matrix(feats, ids):
    """Build the input table for a classifier: one row per id, with all feature sets side by side.

    feats: list of dictionaries from load_feature, e.g. [dinov2_l, quality]
    ids:   the image_ids wanted, in the wanted order
    Example: dinov2_l (2048 numbers) + quality (74) -> each row has 2048 + 74 = 2122 numbers.
    np.stack makes one table per feature set (rows = ids); np.hstack puts the tables next to each other.
    """
    return np.hstack([np.stack([f[i] for i in ids]) for f in feats])


# ----------------------------------------------------------------------------------------------------------
# The classifiers
# ----------------------------------------------------------------------------------------------------------
def models():
    """Name -> a function that builds a NEW, untrained model.

    Functions (lambda: ...) rather than ready models, because every CV fold must start from scratch: reusing a
    model fitted on another fold would carry over what it learned. See the header for what each model is.
    make_pipeline(StandardScaler(), X) = "rescale the features, then run X", fitted together on the same data.
    """
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=SEED)),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED),
        "mlp": lambda: MLPClassifier(seed=SEED),
        "mlp_h64": lambda: MLPClassifier(hidden=64, seed=SEED),
        "mlp_strong": lambda: MLPClassifier(in_drop=0.5, wd=0.1, seed=SEED),
    }


def best_threshold_f1(y, p):
    """Try thresholds 0.10, 0.11, ..., 0.90 on probabilities p; return (best F1, the threshold that gave it).

    p >= t turns probabilities into 0/1 predictions. Choosing t on the same data it is scored on is optimistic,
    so this is only used as a check (see "tuned" in the header).
    """
    ts = np.linspace(0.1, 0.9, 81)          # 81 evenly spaced thresholds, step 0.01
    f = [f1_score(y, p >= t) for t in ts]
    return max(f), ts[int(np.argmax(f))]


# ----------------------------------------------------------------------------------------------------------
# The experiment loop
# ----------------------------------------------------------------------------------------------------------
def evaluate(feature_sets, model_names=None, out_csv="baseline.csv"):
    """Score every (feature set, model) pair with dev F1 and CV F1; save the table and the OOF predictions.

    feature_sets: {name shown in the results: list of feature files}, e.g. {"eva02_l+quality": ["eva02_l", "quality"]}
    model_names:  only run these models (None = all in models())
    out_csv:      results file name inside logs/
    """
    tr, va = load_labels()
    # train + validation joined, in this order (train rows first). The OOF files use the same order.
    allids = np.concatenate([tr.image_id, va.image_id])
    yall = np.concatenate([tr.artifact_present, va.artifact_present])   # the 941 labels
    rows = []                                                           # one results row per (set, model)
    for fname, names in feature_sets.items():
        feats = [load_feature(n) for n in names]
        # The same features arranged three ways: train only, validation only, train + validation.
        Xtr, Xva, Xall = matrix(feats, tr.image_id), matrix(feats, va.image_id), matrix(feats, allids)
        for mname, make in models().items():
            if model_names and mname not in model_names:
                continue

            # ---- dev F1: train on train.csv, predict validation.csv ----
            # make() builds a fresh model; .fit learns from the training images and labels;
            # .predict_proba returns [P(clean), P(artifact)] per image; [:, 1] keeps P(artifact);
            # ">= 0.5" turns it into 0/1; f1_score compares with the true validation labels.
            dev = f1_score(va.artifact_present, make().fit(Xtr, tr.artifact_present).predict_proba(Xva)[:, 1] >= 0.5)

            # ---- CV F1: 5 folds x 3 repeats on all 941 labelled images ----
            # RepeatedStratifiedKFold yields 15 (train indices a, held-out indices b) pairs: repeat 1's 5 folds,
            # then repeat 2's, then repeat 3's. random_state=SEED makes the splits identical for every model
            # and feature set, so all models are compared on exactly the same splits.
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
            oof = np.zeros((3, len(yall)))   # one row of 941 OOF probabilities per repeat
            for k, (a, b) in enumerate(cv.split(Xall, yall)):
                # k = 0..14; k // 5 = which repeat (0, 1 or 2). Fit on fold k's training part, predict its
                # held-out part b, and store those probabilities at positions b of that repeat's row.
                oof[k // 5, b] = make().fit(Xall[a], yall[a]).predict_proba(Xall[b])[:, 1]
            # F1 at threshold 0.5 for each repeat separately -> mean and spread over the 3 repeats.
            f05 = [f1_score(yall, o >= 0.5) for o in oof]
            # "tuned": best threshold on the OOF probabilities averaged over the repeats (optimistic check).
            tuned, t = best_threshold_f1(yall, oof.mean(0))

            rows.append(dict(features=fname, model=mname, dev_f1=dev, cv_f1=np.mean(f05), cv_sd=np.std(f05),
                             cv_tuned=tuned, thr=t))
            # One line per model as it finishes, e.g.
            # "eva02_l+quality   svm_rbf  dev 0.889  cv 0.886±0.003  tuned 0.891@0.47"
            print(f"{fname:28s} {mname:8s} dev {dev:.3f}  cv {np.mean(f05):.3f}±{np.std(f05):.3f}  "
                  f"tuned {tuned:.3f}@{t:.2f}", flush=True)
            # Save the averaged OOF probabilities for error analysis (review_mistakes.py) and ensembling checks.
            np.save(FEAT / f"oof_{fname}_{mname}.npy", oof.mean(0))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "logs" / out_csv, index=False)
    return out


# ----------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    # With arguments: python baseline.py <out.csv> <model,model> <set> [<set> ...]   (set = feat+feat)
    #   sys.argv[1] = results file name, sys.argv[2] = comma-separated model names, sys.argv[3:] = feature sets.
    #   Each set "a+b" becomes {"a+b": ["a", "b"]}.
    if len(sys.argv) > 1:
        evaluate({s: s.split("+") for s in sys.argv[3:]}, sys.argv[2].split(","), sys.argv[1])
        sys.exit()

    # Without arguments: the standard grid, built from the feature files that exist.
    have = {p.stem for p in FEAT.glob("*.npz")}          # names of the saved feature files
    # Each feature file on its own ...
    sets = {n: [n] for n in ["quality", "dinov2_l", "siglip_so400m", "eva02_l"] if n in have}
    # ... each network together with the quality numbers ...
    for emb in ["dinov2_l", "siglip_so400m", "eva02_l"]:
        if emb in have and "quality" in have:
            sets[f"{emb}+quality"] = [emb, "quality"]
    # ... and all three networks + quality (the run1 features: 2048 + 1152 + 768 + 74 = 4042 numbers per image).
    sets["all3+quality"] = ["dinov2_l", "siglip_so400m", "eva02_l", "quality"]
    # Reference floor: F1 of predicting 1 for all 118 validation images (prints 0.703).
    print("majority-1 baseline F1 on validation:",
          round(f1_score(load_labels()[1].artifact_present, np.ones(118)), 3))
    evaluate(sets)
