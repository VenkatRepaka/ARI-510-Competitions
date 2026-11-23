# Inline Image Classifier v2 — Camera Artifact Detection

Code and results for the private Kaggle class competition
[inline-image-classifier-v-2](https://www.kaggle.com/competitions/inline-image-classifier-v-2).

**Task.** Predict whether a photo has a camera artifact: `artifact_present = 1` if it shows at least one of
physical obstruction, exposure fault, lens contamination, optical/motion blur, or lens fog/condensation,
otherwise 0.
**Metric.** Binary F1 for the positive class (artifact present). Submissions are 0/1, not probabilities.

**Best result.** Frozen embeddings from three pretrained vision models plus 74 hand-crafted image-quality
statistics, classified by an RBF SVM: dev F1 **0.906**, CV F1 **0.886** (`submission_run1_all3q_svm.csv`).

---

## 1. Data

| File | Rows | Columns |
|---|---|---|
| `data/train.csv` | 823 images (448 with artifact, 54 %) | `image_id, filename, artifact_present` |
| `data/validation.csv` | 118 images (64 with artifact, 54 %) | `image_id, filename, artifact_present` |
| `data/test.csv` | 236 images | `image_id, filename` |
| `data/sample_submission.csv` | 236 | `image_id, artifact_present` |

- Images are a mix of phone photos and web images (jpg, jpeg, png, webp; HEIC/AVIF were converted to PNG by
  the host). Sizes vary widely.
- Data licence: MIT (stated on the competition page).
- Host rules: use train for fitting and validation for development; do not label test images by hand; treat
  `image_id` and filenames as identifiers, not features; load as RGB and honour EXIF orientation.
- **Hard negatives.** The clean class includes weather fog, raindrops on a window that are in focus, shallow
  depth of field, and dark night scenes. These look like artifacts at a low level, so the model needs to
  understand *what* the image shows, not only how sharp or bright it is.
- Predicting 1 for every image already gives F1 0.703 on validation, so that is the floor.

**Preprocessing.** Every image is loaded with PIL, rotated by its EXIF orientation and converted to RGB. No
other cleaning. Each feature script then resizes the image for its own needs (see section 3).

---

## 2. How models are compared

Two scores are reported for every model. Both use a threshold of 0.5 on the predicted probability.

- **dev F1:** fit on `train.csv`, score on `validation.csv` (118 images). This is the host's development
  split. With 118 images one image changes F1 by about 0.01, so it is noisy.
- **CV F1:** 5-fold stratified cross-validation repeated 3 times (`RepeatedStratifiedKFold(n_splits=5,
  n_repeats=3, random_state=0)`) on train + validation together (941 images). The mean over the 3 repeats is
  reported with its standard deviation. This is the more reliable number and was used to choose between models.
- **tuned:** the best single threshold on the pooled out-of-fold probabilities. It is optimistic (the
  threshold is chosen on the same data) and was only used to check that 0.5 is close to optimal. It was
  (0.41–0.48 for the good models), so all submissions use 0.5.

All fitting (scaling, models) happens inside each fold. All seeds are fixed (0), the MLP runs on CPU with a
seeded generator, and every script gives the same numbers when rerun. The test set is never used for anything
except the final prediction.

### How the approaches are organised

There is one simple idea behind every experiment:

> **Every approach = some feature files + one classifier.**
> The code is the same for all of them. Only the list of feature files (and the classifier name) changes.

**The feature files are the ingredients.** Each feature script computes numbers for every image once and saves them
in `features/`. Nothing after that looks at the images again.

| Feature file (name used in commands) | Numbers per image | What it is | Made by |
|---|---|---|---|
| `quality` | 74 | hand-made quality statistics (sharpness, exposure, haze, …) | `quality_features.py` |
| `dinov2_l` | 2,048 | DINOv2 embedding | `embed_features.py` |
| `siglip_so400m` | 1,152 | SigLIP embedding | `embed_features.py` |
| `eva02_l` | 768 | EVA02 embedding | `embed_features.py` |
| `tilesim_eva02_l`, `tileagg_eva02_l` | 19 / 2,323 | summaries of EVA02 embeddings of 9 image tiles | `tile_embed.py`, then `tile_features.py` |
| `bands` | 50 | quality statistics on the top / middle / bottom thirds | `band_features.py` |

**Each approach picks some ingredients and a classifier:**

| Approach | Feature files | Classifier | dev F1 | CV F1 | Submitted |
|---|---|---|---|---|---|
| ALL-ONES (reference) | none: predicts 1 for every image | none | 0.703 | – | no |
| QUALITY-STATS | `quality` | `svm_rbf` | 0.776 | 0.780 | no |
| SINGLE-BACKBONE | `dinov2_l` *or* `siglip_so400m` *or* `eva02_l` | `svm_rbf` | 0.790 / 0.898 / 0.873 | 0.853 / 0.882 / 0.880 | no |
| BACKBONE+QUALITY | one backbone + `quality` (best: `eva02_l` + `quality`) | `svm_rbf` | 0.889 | 0.886 | no |
| **ALL3-SVM** | `dinov2_l` + `siglip_so400m` + `eva02_l` + `quality` | `svm_rbf` | **0.906** | **0.886** | **run1** |
| **MLP** | `eva02_l` + `quality` | `mlp_strong` | 0.873 | 0.880 | **run2** |
| REGION-FOG-FEATURES | best set above + `tilesim_eva02_l` / `tileagg_eva02_l` / `bands` | `svm_rbf` | 0.857–0.906 | 0.867–0.885 | no |

**You choose the ingredients on the command line**: the words after the model name.

```
make_submission.py   svm_rbf     dinov2_l siglip_so400m eva02_l quality
                     └ model ┘   └──────────── feature files ───────────┘
```

Change those words and you have changed the approach. In `baseline.py` a set of files is written with `+`
between them (`eva02_l+quality`); in `make_submission.py` with spaces.

**Where an approach is trained.** No approach has a file of its own. All of them are trained by the same two
lines in `baseline.py`, inside `evaluate()`. For example, QUALITY-STATS goes through these steps:

| Step | Where in `baseline.py` | What happens |
|---|---|---|
| 1. The set is named | bottom of the file (`sets = {n: [n] for n in ["quality", …]}`) | creates the set `"quality"` |
| 2. The files are loaded | `evaluate()`: `feats = [load_feature(n) for n in names]` | opens `features/quality.npz` (74 numbers per image) |
| 3. The classifier recipe | `models()`: `"svm_rbf"` | scale the numbers, then an RBF SVM with C = 3 |
| **4. Trained for the dev score** | `evaluate()`: `dev = f1_score(… make().fit(Xtr, …) …)` | **trained on train.csv** (823 images), predicts validation.csv |
| **5. Trained for the CV score** | `evaluate()`: `oof[k // 5, b] = make().fit(Xall[a], …) …` | **trained 15 more times** (5 folds × 3 repeats) |
| 6. Result saved | `evaluate()`: `out.to_csv(…)` | one row in `logs/baseline.csv` |

Steps 4 and 5 are the only places where a model learns during the experiments. For the two submitted runs,
`make_submission.py` trains the chosen model once more on all 941 labelled images and predicts the test set.

**Where to find an approach's results.** Every results table in `logs/` has a `features` column that lists the
ingredients joined with `+`, and a `model` column:

```
features          model     dev_f1   cv_f1
eva02_l+quality   svm_rbf   0.889    0.886
```

| Results file | Approaches in it |
|---|---|
| `logs/baseline.csv` | QUALITY-STATS, SINGLE-BACKBONE, BACKBONE+QUALITY, ALL3-SVM (plus logistic regression and gradient boosting on the same sets) |
| `logs/mlp.csv` | MLP |
| `logs/tiles_eva.csv`, `logs/bands.csv` | REGION-FOG-FEATURES |

The out-of-fold predictions of each row are saved under the same name: `features/oof_<features>_<model>.npy`.

**To rerun one approach on its own**, name its classifier and feature set (results go to `logs/check.csv`, so
the original tables are not overwritten):

```bash
../.venv/bin/python baseline.py check.csv svm_rbf quality
```

---

## 3. Files, in the order they were built

### 3.1 `quality_features.py` — hand-crafted image-quality statistics

Each image is downscaled so its long side is 1024 px. The following statistics are computed on the whole image
and on each cell of a 3×3 grid (the grid gives min / max / spread across cells, plus border-minus-centre,
because obstructions and smears often cover only part of the frame):

- **Sharpness / blur:** variance of the Laplacian, gradient mean, 99th percentile and Tenengrad, edge density
  (Canny), gradient anisotropy (motion blur smears in one direction).
- **Frequency content:** share of the spectrum in high and middle frequencies.
- **Exposure:** brightness mean, std and percentiles, share of clipped highlights (≥ 250) and shadows (≤ 5),
  dynamic range, histogram entropy.
- **Colour and haze:** saturation mean and std, dark-channel mean and 10th percentile (a standard haze cue),
  RMS contrast, local contrast.
- Aspect ratio.

74 features in all. Output: `features/quality.npz`.

```bash
../.venv/bin/python quality_features.py
```

### 3.2 `embed_features.py` — frozen pretrained image embeddings

Only the image towers are used, with frozen weights. Each image is encoded twice, as is and mirrored, and the
two embeddings are averaged. All embeddings are L2-normalised.

| Key | Model | Library / weights | Input | Dim |
|---|---|---|---|---|
| `dinov2_l` | DINOv2 ViT-L/14 | timm `vit_large_patch14_dinov2.lvd142m` | 448 px; CLS token + mean patch token | 2048 |
| `siglip_so400m` | SigLIP SO400M/14 | open_clip `ViT-SO400M-14-SigLIP-384`, `webli` | 384 px | 1152 |
| `eva02_l` | EVA02-L/14 (CLIP) | open_clip `EVA02-L-14-336`, `merged2b_s6b_b61k` | 336 px | 768 |

Each model takes about 7–11 minutes for all 1,177 images on an Apple Silicon GPU (MPS).
Output: `features/<key>.npz` (`ids, split, img, img_flip`).

```bash
../.venv/bin/python embed_features.py
```

### 3.3 `baseline.py` — classical models on the features

Models: logistic regression (C = 0.1, standardised), RBF SVM (C = 3, standardised, Platt probabilities),
HistGradientBoosting (300 iterations, learning rate 0.05). Feature sets are concatenated with `+`.

```bash
../.venv/bin/python baseline.py
```
```bash
../.venv/bin/python baseline.py mlp.csv mlp,mlp_strong eva02_l+quality siglip_so400m+quality
```

With no arguments it runs every single feature set, each backbone + quality, and all three backbones + quality.
Results go to `logs/baseline.csv` (or the named CSV), and the out-of-fold probabilities to
`features/oof_<features>_<model>.npy`.

| Features | logreg dev / CV | **RBF SVM dev / CV** | HGB dev / CV |
|---|---|---|---|
| quality (74 stats) | 0.776 / 0.758 | 0.776 / 0.780 | 0.767 / 0.772 |
| DINOv2-L | 0.773 / 0.827 | 0.790 / 0.853 | 0.824 / 0.836 |
| SigLIP SO400M | 0.866 / 0.855 | 0.898 / 0.882 | 0.855 / 0.854 |
| EVA02-L | 0.848 / 0.845 | 0.873 / 0.880 | 0.878 / 0.848 |
| DINOv2-L + quality | 0.783 / 0.831 | 0.833 / 0.859 | 0.810 / 0.841 |
| SigLIP + quality | 0.864 / 0.854 | 0.915 / 0.880 | 0.875 / 0.861 |
| EVA02 + quality | 0.816 / 0.844 | 0.889 / 0.886 | 0.887 / 0.857 |
| **all 3 backbones + quality** | 0.848 / 0.867 | **0.906 / 0.886 ± 0.003** | 0.816 / 0.866 |

Findings:

- The CLIP-style models (SigLIP, EVA02), trained on image–text pairs, beat DINOv2 by about 0.03 CV. Their
  embeddings seem to encode concepts like "foggy", "blurry" or "through a window", which is what the labels need.
- The quality statistics alone reach 0.78, and adding them to an embedding helps a little (+0.006 for EVA02).
- The RBF SVM is the best classifier on every feature set. The three-backbone SVM ties EVA02 + quality on CV
  and has the better dev score, so it was chosen for run1.

### 3.4 `mlp.py` — neural network (PyTorch MLP)

A small network trained on the same frozen features:

- Per network: standardise → input dropout → Linear(D, 256) → GELU → dropout 0.5 → Linear(256, 1).
- 5 networks with different random initialisations are trained side by side as one batched module (`torch.bmm`)
  and their probabilities are averaged. The loss is summed over the networks, so each network only learns from
  its own weights. Training all five together takes 14 s per fit instead of about 60 s in a loop.
- AdamW (lr 1e-3), cosine learning-rate schedule, batch 64, a **fixed** 60 epochs. There is no early stopping,
  so held-out data is never used during training.
- Variants in `baseline.py`: `mlp` (input dropout 0.2, weight decay 0.01), `mlp_h64` (64 hidden units),
  `mlp_strong` (input dropout 0.5, weight decay 0.1).

Results (`logs/mlp.csv`):

| Features | mlp dev / CV | mlp_h64 dev / CV | mlp_strong dev / CV |
|---|---|---|---|
| SigLIP + quality | 0.882 / 0.875 | 0.882 / 0.875 | 0.898 / 0.876 |
| EVA02 + quality | 0.864 / 0.871 | 0.864 / 0.871 | **0.873 / 0.880** |
| all 3 + quality | 0.857 / 0.877 | 0.873 / 0.877 | 0.857 / 0.877 |

The best MLP (EVA02 + quality, `mlp_strong`) is 0.006 below the SVM on CV. Averaging the SVM's and the MLP's
out-of-fold probabilities gave 0.883–0.893 against 0.890 for the SVM alone, which is within noise. With about
750 training images per fold, the SVM's simpler decision rule generalises at least as well as a network.
The best MLP was written as run2.

### 3.5 `review_mistakes.py` — what the best model gets wrong

```bash
../.venv/bin/python review_mistakes.py all3+quality_svm_rbf
```

It reads run1's out-of-fold probabilities and writes `logs/errors_<name>.csv` plus contact sheets of the 36 most
confident misses (`_FN.jpg`) and false alarms (`_FP.jpg`).

- Run1 makes **61 misses and 50 false alarms out of 941** images (on validation: 9 and 7).
- **Fog and mist** are the main confusion. Misty landscapes appear among both misses and false alarms and look
  almost the same. The label seems to depend on whether the fog is *on the lens* or *in the scene*.
- **Raindrops on windows** come second: some are labelled artifact (drops on the glass in front of the
  camera), others clean.
- **Label conflicts.** A near-duplicate search (DINOv2 cosine > 0.95, `logs/near_duplicates.csv`) found 14 pairs.
  Three pairs are the same image with **opposite labels**:

  | Image A (label) | Image B (label) | Content |
  |---|---|---|
  | 0446 train (0) | 0403 validation (1) | rainy window |
  | 0136 train (0) | 0087 train (1) | teal misty forest |
  | 0067 train (1) | 0135 train (0) | foggy field |

  No model can get both copies of a pair right, so these set a small ceiling on F1. Image 0403 is in
  validation, so one dev error cannot be avoided. The labels were left as they are: changing them would mean
  deciding labels by eye. `logs/conflicting_duplicates.jpg` shows them.
- Two training images are near-copies of test images (0047 ≈ test 0070, 0200 ≈ test 0239). Nothing was done
  about this; the test images were not looked at or labelled.

### 3.6 `tile_embed.py`, `tile_features.py`, `band_features.py` — trying to separate lens fog from weather fog

**Idea:** lens fog or a smear veils the whole frame evenly, nearby objects included. Weather fog grows with
distance, so the bottom of the frame (near the camera) stays sharp and contrasty while the top fades. Comparing
regions of the image might capture this.

- **`tile_embed.py`** embeds a 3×3 grid of overlapping tiles (each 40 % of the width and height) with EVA02-L
  (about 26 minutes). Output: `features/tiles_eva02_l.npz` (N × 9 × 768).
- **`tile_features.py`** turns the tiles into
  - `tilesim_eva02_l`: 19 scalars on how alike the tiles are (pairwise cosine mean/min/std, top row vs bottom
    row, centre vs border, left vs right, each tile vs the full image, neighbouring tiles);
  - `tileagg_eva02_l`: those scalars plus the mean tile embedding, the per-dimension spread over tiles, and the
    top-row minus bottom-row embedding.
- **`band_features.py`** computes 10 quality statistics on the top, middle and bottom thirds of the image and
  their differences. Output: `features/bands.npz`.

Results (RBF SVM, `logs/tiles_eva.csv`, `logs/bands.csv`):

| Features | dev F1 | CV F1 |
|---|---|---|
| tile similarity scalars alone | 0.680 | 0.703 |
| band statistics alone | 0.721 | 0.738 |
| EVA02 + quality (reference) | 0.889 | 0.886 |
| EVA02 + quality + tile similarity | 0.889 | 0.884 |
| EVA02 + quality + tile aggregates | 0.857 | 0.867 |
| run1 features (reference) | 0.906 | 0.886 |
| run1 features + bands | 0.906 | 0.885 |
| run1 features + tile aggregates | 0.866 | 0.875 |

A logistic-regression stack of run1's out-of-fold probability with the tile and band scalars also did not
help (0.879–0.887 against 0.893 for run1 at its tuned threshold). The new features carry some signal on their
own, but nothing that the full-image embeddings do not already have, and the large tile-aggregate vectors add
noise. **No gain**, so run1 stayed the best model and no new submission was made from this step.

### 3.7 `make_submission.py` — final predictions

Refits one model on train + validation (941 images), predicts the 236 test images, thresholds at 0.5, checks
that the row order matches `sample_submission.csv`, and writes the CSV.

The first argument is the model name (from `baseline.py`); every following word is a **saved feature file**
in `features/`, written without `.npz`. Nothing is recomputed from the images; the script loads these files:

| Argument | File | Numbers per image | Saved by |
|---|---|---|---|
| `quality` | `features/quality.npz` | 74 | `quality_features.py` (hand-made quality numbers) |
| `dinov2_l` | `features/dinov2_l.npz` | 2,048 | `embed_features.py` (DINOv2 embedding) |
| `siglip_so400m` | `features/siglip_so400m.npz` | 1,152 | `embed_features.py` (SigLIP embedding) |
| `eva02_l` | `features/eva02_l.npz` | 768 | `embed_features.py` (EVA02 embedding) |

So `quality_features.py` and `embed_features.py` must have been run once before (they already have been).

```bash
../.venv/bin/python make_submission.py svm_rbf dinov2_l siglip_so400m eva02_l quality --out submission_run1_all3q_svm.csv
```
```bash
../.venv/bin/python make_submission.py mlp_strong eva02_l quality --out submission_run2_eva02q_mlp.csv
```

Both commands give exactly the same file on every run (run1 md5 `756e763a…`).

---

## 4. Results

| Submission file | Approach | dev F1 | CV F1 | Test positives | Public F1 |
|---|---|---|---|---|---|
| `submission_run1_all3q_svm.csv` | DINOv2-L + SigLIP + EVA02-L embeddings + 74 quality stats, RBF SVM (C = 3) | **0.906** | **0.886 ± 0.003** | 133 / 236 (56 %) | not yet submitted |
| `submission_run2_eva02q_mlp.csv` | EVA02-L embeddings + 74 quality stats, PyTorch MLP ensemble (`mlp_strong`) | 0.873 | 0.880 ± 0.002 | 125 / 236 (53 %) | not yet submitted |

Run1 and run2 agree on 91 % of the test images. Reference: predicting 1 everywhere gives F1 0.703 on validation.

**Final selection:** run1, chosen on CV and dev F1 (not on the public score).

---

## 5. What worked and why

1. **Pretrained image–text models carry the signal.** The difference between an artifact and a hard negative
   (lens fog vs weather fog, drops on the lens vs a rainy window seen through glass) is about meaning more than
   pixel statistics. SigLIP and EVA02 embeddings alone reach 0.88 CV; the hand-crafted statistics alone 0.78.
2. **Simple classifiers on frozen features.** With fewer than 1,000 labelled images, an RBF SVM on frozen
   embeddings beat gradient boosting, logistic regression and an MLP. Fine-tuning a backbone was not tried
   because the frozen-feature models were already close to the label-noise ceiling.
3. **The labels limit the score.** Identical images carry opposite labels, and fog/raindrop scenes are labelled
   both ways. The remaining errors are mostly cases a human would also find ambiguous.
4. **Extra features found by error analysis did not help.** Region-based fog features had some signal on their
   own but added nothing on top of the full-image embeddings.

---

## 6. Folder map

```
data/                         competition data (CSV files + images/{train,validation,test}), not in git
features/
  quality.npz                   74 quality statistics           (quality_features.py)
  dinov2_l.npz, siglip_so400m.npz, eva02_l.npz   frozen embeddings (embed_features.py)
  tiles_eva02_l.npz             3x3 tile embeddings             (tile_embed.py)
  tilesim_eva02_l.npz, tileagg_eva02_l.npz       tile summaries  (tile_features.py)
  bands.npz                     top/middle/bottom band stats    (band_features.py)
  oof_<features>_<model>.npy    out-of-fold probabilities, mean of 3 repeats (baseline.py)
logs/
  baseline.csv, mlp.csv, tiles_eva.csv, bands.csv   result tables
  errors_all3+quality_svm_rbf.csv, *_FN.jpg, *_FP.jpg   error review
  near_duplicates.csv, conflicting_duplicates.jpg      duplicate check
  sheet_label0.jpg, sheet_label1.jpg                   sample images of each class
  embed.log, quality.log, mlp.log, tiles.log           run logs
quality_features.py, embed_features.py, tile_embed.py, tile_features.py, band_features.py   feature scripts
baseline.py, mlp.py           models and evaluation
review_mistakes.py            error analysis
make_submission.py            final predictions
submission_run*.csv           submitted files (see section 4)
```

## 7. Software and licences

- Python 3.13 with numpy, pandas, scikit-learn 1.9, OpenCV 5.0, Pillow.
- PyTorch 2.15 (MPS on Apple Silicon for embeddings; CPU for the MLP), timm 1.0, open_clip_torch 4.2.
- Shared virtual environment: `../.venv`.

External models: DINOv2 (Apache-2.0), SigLIP SO400M (Apache-2.0), EVA02-CLIP (MIT), all used with frozen
weights. No external data was used. Competition data: MIT.
