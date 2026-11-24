# Chore Completion Classifier

Code and results for the private Kaggle class competition
[chore-completion-classifier](https://www.kaggle.com/competitions/chore-completion-classifier).

**Task.** Each photo shows one of two chores, washing dishes or taking out the trash. Predict the chore and
whether it is finished. That gives 4 labels: `Washing Dishes|Complete`, `Washing Dishes|Incomplete`,
`Taking Out Trash|Complete`, `Taking Out Trash|Incomplete`.
**Metric.** Macro F1 over the 4 classes, higher is better.

**Best public result.** Extra trees on frozen embeddings from 5 image backbones, trained on the 66 original
photos: public macro F1 **0.94155** (`submission_public0.94155_extratrees.csv`, frozen in `best_0.94155/`).
**Best cross-validation result.** DINOv2 RBF SVM + overflow-tile and open-bag scores, with a dishes-only
logistic regression routed in: CV macro F1 **0.916** (`submission_dinov2_open_routed.csv`, public 0.88311).

> Status: this document is being written up from the experiment logs. Sections marked *TODO* still need
> numbers checked or text filled in.

---

## 1. Data

| File | Rows | Columns |
|---|---|---|
| `data/train_original.csv` | 66 photos (the competition training set) | `id, image_file, image_url, label` |
| `data/train.csv` | 66 originals + 19 external web photos (ids 9NN) | same |
| `data/test.csv` | 36 photos | `id, image_file, image_url` |
| `data/images/` | all competition + external images, `image_NNN.jpg` | |

- Class counts in the 66 originals: Trash Incomplete 20, Trash Complete 13, Dishes Complete 17,
  Dishes Incomplete 16. Classes are close to balanced; every model uses `class_weight="balanced"`.
- **Download.** Kaggle "Download All" did not save a file, so the CSVs were rebuilt from the Kaggle Data
  Explorer. The images are fetched from the `image_url` column (a public GitHub repository). That repository
  also holds labels; they are **never used**, since that would leak the test set.
- The competition photos are Unsplash photos. Any new web image must be checked against all 102 competition
  photos (DINOv2 cosine) before use: several web candidates turned out to be the same photo as a train or
  test image (#911 = train #12, #921 = test #27, #922 = test #30, #912 = same sink as test #23).
- Older versions of the training file are kept for reproducibility: `train_with15_external.csv`,
  `train_sinks.csv`, `train_12ext.csv`, `train_21ext.csv`. Scripts pick a file with the `TRAIN_CSV`
  environment variable.

**What the labels mean** (from looking at the training photos, `error_analysis.py`):

- *Dishes Complete*: clean dishes on a rack, in a dishwasher, put away, or an empty sink.
- *Dishes Incomplete*: food residue, dishes waiting to be washed, someone still washing.
- *Trash Complete*: the trash is contained: tied bags, closed or lined-up bins, sorted recycling.
- *Trash Incomplete*: overflowing bins, open bags, loose litter.

The chore type is never wrong in any model. **Every error is Complete vs Incomplete within one chore**, and
the hard cases are wide or dark scenes where the cue is small (#2, #46, #60, #99, #101). A few dishes labels
look inconsistent (#31, #60 are labelled Incomplete but look like clean dishes; #101, #64, #40 are labelled
Complete with dishes in the sink).

### External images

All external images are in `../external_images/` with source and licence in `../external_images/CREDITS.csv`.
Only sources with a stated licence are used (Unsplash, Pixabay, Wikimedia Commons / Flickr CC licences).

| Group | ids | Label | Used in current `train.csv` |
|---|---|---|---|
| Tied black / white bags | 901–905, 907, 910 | Trash Complete | yes |
| Clean sink | 912 | Dishes Complete | yes |
| Overflowing bins (Pixabay, #944 AI-generated) | 941–944 | Trash Incomplete | yes |
| Overflowing dumpsters (Unsplash) | 945, 947 | Trash Incomplete | yes |
| Open bags (Unsplash) | 951–955 | Trash Incomplete | yes |
| Dropped | 906, 908, 909, 911, 913–924, 946, 948, 949 | | no (duplicates, or flipped train #13) |

External rows are only ever added to **training folds**. CV is always scored on the 66 originals.

The **CDCM** dataset (Kaggle `rodrigolaguna/clean-dirty-containers`, in `../external_images/cdcm/`) is used
only to train an auxiliary "dirty bin" classifier (section 3.6), not as training photos.

---

## 2. How models are compared

- **Repeated stratified 5-fold CV on the 66 original photos**, `random_state=0`. The comparison notebooks use
  3 repeats; the later scripts (`aux_cv.py` onward) use 10 repeats.
- **Grouped by photo.** All views of one photo (mirror, crops, augmentations, tiles) stay in the same fold.
  An early "augment, then shuffle" dev split put views of the same photo on both sides and gave a leaky
  0.995 (`dev_predictions.ipynb`); it is not used.
- Everything that learns (scalers, PCA, classifiers, the auxiliary tile classifiers) is fitted inside each
  fold.
- The score is the mean macro F1 over the repeats. With ~33 photos per chore, one photo flip moves F1 by
  about 0.03, and the repeat-to-repeat spread is ±0.02–0.03. **Differences below ~0.02 are noise.**
- Per-photo "times wrong out of 10 repeats" tables are used to see which photos a change fixes or breaks.
- Many configurations were tried, so the best CV numbers are optimistic (winner's curse). `per_chore.ipynb`
  measures this with a nested selection (section 3.9).

**Host benchmarks (public board):** trained 0.74743, random 0.32619, majority 0.12500.

---

## 3. Files, in the order they were built

### 3.1 `chore_classifier.ipynb` — first baseline (CLIP)

Downloads the photos, explores the classes, and fits a linear probe on CLIP ViT-H-14 image embeddings,
blended 0.7 / 0.3 with CLIP zero-shot text prompts. **CV macro F1 0.788.** After this the user decided to use
**image-only models** (no text encoder, no language model).

### 3.2 `extract_features.py` + `train.py` — frozen backbones and the model grid

- `extract_features.py` embeds every image with 5 frozen backbones and caches them in `features/*.npz`:

  | key | model | weights |
  |---|---|---|
  | `dinov2_l` | `vit_large_patch14_dinov2.lvd142m` (timm) | LVD-142M |
  | `eva02_l` | OpenCLIP `EVA02-L-14-336` | `merged2b_s6b_b61k` |
  | `siglip_so400m` | OpenCLIP `ViT-SO400M-14-SigLIP-384` | `webli` |
  | `vith14_laion` | OpenCLIP `ViT-H-14` | `laion2b_s32b_b79k` |
  | `vitl14_openai` | OpenCLIP `ViT-L-14` | `openai` |

  Each photo gets 7 views (image, mirror, crops); `--aug N` adds N random augmented views, `--color` adds
  colour-changed views.
- `train.py` scores every (backbone, view mode, classifier) with CV, then ensembles. View modes: `full`
  (mean of image + mirror), `mean7`, `aug`, `aug+tta`. Classifiers: logistic regression, linear / RBF SVM,
  kNN, LDA, ridge, extra trees, MLP, label spreading. Logs in `logs/train_*.log`, `logs/results_*.csv`.
- `_clf.py` holds shared classifier helpers.

**Result:** extra trees on all 5 backbones, `full` view → CV 0.812, **public 0.94155** (best public score).
LR on 20 augmented DINOv2 views (`submission_aug_dinov2.csv`) → public 0.883.

```bash
python extract_features.py dinov2_l eva02_l siglip_so400m vith14_laion vitl14_openai
python extract_features.py --aug 20 dinov2_l
python train.py --repeats 10
```

### 3.3 `best_0.94155/` — frozen copy of the best public submission

Exact copies of `train.py`, `extract_features.py` and the 66-row training file, verified to reproduce the
0.94155 file bit for bit. See `best_0.94155/README.md` for the command; features are in
`features_backup_before_external/`.

### 3.4 Error analysis and patch-level models

- `error_analysis.py` → `logs/error_analysis.csv`, contact sheets `logs/error_sheet_*.jpg`. Defines the label
  meanings in section 1.
- `extract_patches.py` + `patch_models.py`: DINOv2 patch tokens (`features/dinov2_l_patches.npz`), chore first
  then Complete / Incomplete. Pooled [CLS, mean, max] LR 0.814 ≈ baseline; patch top-k LR ≤ 0.75, NBNN ≤ 0.57.
  **No gain with 66 photos.**
- `dishes_experiment.py`: dishes-only model plus web clean / dirty sinks (`train_sinks.csv`). Best 0.840.

### 3.5 Model comparison notebooks

Same protocol (66 originals, 3×5 photo-grouped CV, 6 models incl. a PyTorch MLP, view modes
full / mean7 / fixed7 / aug / fixed7+aug / fixed7+color / fixed7+color+aug). Each notebook writes a
`*_best.csv` (best single model) and `*_crops.csv` submission.

| Notebook | Training data | Best single model | CV |
|---|---|---|---|
| `model_comparison.ipynb` | 66 originals | SigLIP aug, LR | 0.848 |
| `model_comparison_external.ipynb` | + 11 external | DINOv2 aug, RBF SVM | 0.856 |
| `model_comparison_overflow.ipynb` | + 15 external (adds overflowing bins) | 5-backbone concat, MLP | 0.862 (ensemble 0.870) |
| `model_comparison_12ext.ipynb` | + 12 external | 5-backbone concat, MLP | 0.862 (ensemble 0.871) |

External web photos raised CV, but `submission_dinov2_svm_external.csv` (CV 0.845) scored only **0.88311**
public: tied-bag web photos taught "black bag visible = Complete" and flipped an overflowing dumpster.

### 3.6 Auxiliary "dirty bin" scores — `overflow_aux.py`, `aux_cv.py`

- `overflow_aux.py` trains an LR on 300 + 300 CDCM smartphone images (clean vs dirty container; held-out
  accuracy SigLIP 0.935, DINOv2 0.895), then scores each competition photo as a whole
  (`features/overflow_score.npz`) and on 2×2 + 3×3 tiles (`features/overflow_tiles.npz`).
- `bag_tiles.py`: per-tile cosine to the external tied-bag vs open-bag images → `open_tile_max/top2`
  (`features/bag_tiles.npz`, AUC 0.79 alone).
- `aux_cv.py` (10 repeats) appends these scores to the embeddings (`--tiles`, `--bags`, `--open`).

| DINOv2 full + … , RBF SVM (10 repeats) | CV |
|---|---|
| nothing (base) | 0.852 |
| whole-image overflow score | 0.875 |
| overflow tiles | 0.890 |
| tiles + open-bag score, 66 + 19 external | **0.912 ± 0.020** |

`make_submission_aux.py` (deterministic) writes `submission_dinov2_tiles_svm.csv` (public 0.88311),
`submission_dinov2_overflow_svm.csv`, `submission_dinov2_tiles_open_svm.csv` (identical to the tiles file on
all 36 test rows) and `submission_dinov2_open_routed.csv`.

```bash
python make_submission_aux.py --open
```

### 3.7 Augmentation experiments

| Script | What | Effect |
|---|---|---|
| `zoom_aug.py` | zoom in 1.25/1.5, out 0.8/0.67 (tiles recomputed) | SVM 0.912 → 0.897; LR up to 0.905 |
| `rand_aug.py` | 60 random crop / rotate / grey / dark views per photo | SVM down; LR +30 views 0.913 (ties SVM) |
| `crop_aug.py` | 19-view crop × transform grid | stopped before finishing (~3 h) |
| `ens_aug.py`, `ens_open.py` | probability averaging of SVM / LR / MLP | ≤ 0.915, noise |

Consistent pattern: **augmentation helps logistic regression and hurts the RBF SVM.** Routing dishes photos
to LR +10 augmented views (`ens_aug.py`) gives the routed model, CV 0.916.

### 3.8 Dishes experiments (all not adopted)

| Script | Idea | Result |
|---|---|---|
| `dishes_errors.py`, `trash_errors.py` | per-photo error sheets | `logs/*_errors.*` |
| `drop_dishes_cv.py` | drop suspect labels (#31, #60, …) from training folds | within noise |
| `relabel_sink_cv.py` | user rule "anything in the sink = Incomplete" | 0.873 vs relabelled targets, but #7, #22 always wrong |
| `dish_boxes.py` + `dish_crops.py` | hand boxes on 33 dishes photos → dirty-dish tile score | hurts (0.912 → 0.85–0.89) |
| `soiled_windows.py` | 74 sliding windows, in-fold dirty detector (AUC 0.893) | hurts (dishes 0.875 → 0.83–0.85) |
| `dishes_cv.py` | dishes-only binary grid, pseudo-labels | ceiling ~4 wrong of 33 |
| `fg_features.py` | foreground-only DINOv2 (k-means on patches) | foreground only 0.767; no net gain |
| `ft_dinov2.py` | LP-FT of DINOv2-L (last 4 blocks, 448 px) — **trained neural net** | 0.856 = frozen SVM on same folds |
| `ablate13.py` | which external photos flip train #13 | dropped #948, #949 |

`DISH_CROP_PLAN.md` is the written plan for the crop experiment.

### 3.9 `per_chore.ipynb` — one binary model per chore

A chore router (DINOv2 LR, perfect in CV), then a separate model grid for dishes and for trash.

| Combination | CV macro F1 |
|---|---|
| 4-class baseline (DINOv2 + tiles RBF SVM) | 0.890 |
| best dishes + best trash (picked from the grid) | 0.894 |
| top-5 average per chore | 0.879 |

Nested selection (choose the model inside each outer fold): dishes 0.878 → 0.899, trash 0.910 → **0.849**.
The trash "best" was mostly luck. `submission_per_chore.csv` (top-5 average) was written, not submitted.

---

## 4. Results

### Submitted

Run numbers follow submission order (*TODO: confirm against the Kaggle submission page*).

| Run | File | Approach | CV F1 | Public F1 |
|---|---|---|---|---|
| run1 | `submission_aug_dinov2.csv` | DINOv2, LR on 20 aug views, TTA | *TODO* | 0.883 |
| run2 | `submission_public0.94155_extratrees.csv` | extra trees, 5 backbones, full view, 66 originals | 0.812 | **0.94155** |
| run3 | `submission_dinov2_svm_external.csv` | DINOv2 RBF SVM C=3 + 15 web photos | 0.845 | 0.88311 |
| run4 | `submission_dinov2_tiles_svm.csv` | DINOv2 + CDCM tile scores, RBF SVM, + 12 web | 0.890 | 0.88311 |
| run5 | `submission_dinov2_open_routed.csv` | run4 + open-bag score + 19 web, dishes → LR +10 aug | **0.916** | 0.88311 |

Baselines: CLIP linear probe + zero-shot CV 0.788; host trained benchmark 0.74743 public.

**Reading the public board.** The public set is about half of the 36 test photos. With 4 classes, 0.94155 and
0.88311 differ by roughly one wrong photo, so the public board cannot separate these models. Final choices are
made from CV, not public score (as the assignment requires).

### Prepared, not submitted

| File | Approach | CV F1 |
|---|---|---|
| `submission_notebook_best.csv` / `_crops.csv` | model_comparison (66 only) | 0.848 |
| `submission_notebook_external_*.csv` | + 11 external | 0.856 |
| `submission_notebook_overflow_*.csv` | + 15 external | 0.860–0.870 |
| `submission_notebook_12ext_*.csv` | + 12 external | 0.862–0.871 |
| `submission_external.csv` | train.py extra trees + external | 0.770 |
| `submission_dinov2_overflow_svm.csv` | DINOv2 + whole-image overflow score | 0.875 |
| `submission_dinov2_tiles_open_svm.csv` | + open-bag score (same predictions as run4) | 0.912 |
| `submission_per_chore.csv` | per-chore top-5 averages | 0.879 |

**Planned final selection:** `submission_dinov2_open_routed.csv` (best CV) and
`submission_public0.94155_extratrees.csv` (best public).

---

## 5. What worked and why

1. **Frozen foundation-model embeddings + a small classifier** are the right size for 66 photos. Fine-tuning
   DINOv2 matched the frozen SVM but did not beat it.
2. **The chore is easy; completeness is the whole problem.** Every error is Complete vs Incomplete.
3. **Trash: local evidence helped.** An auxiliary dirty-bin classifier trained on a separate licensed dataset
   (CDCM), applied to tiles, plus an open-vs-tied bag score, raised trash accuracy: 0.852 → 0.912 CV.
4. **Dishes: nothing beat the plain model.** Crops, detectors, relabelling and per-chore models all stayed at
   ~4 wrong of 33. Part of this looks like label noise (#31, #60, #101).
5. **External photos help CV but can teach shortcuts** (tied black bag = Complete). Every one must be checked
   for duplicates against competition photos and kept in training folds only.
6. **Augmentation helps linear models and hurts the RBF SVM.**

What did not work: patch-level / NBNN models, foreground masks, colour views, zoom / random augmentation for
the SVM, dirty-dish detectors, ensembling, per-chore model picking (nested CV shows the trash gain was luck).

---

## 6. Folder map

```
data/
  train.csv                 current training file (66 originals + 19 external)
  train_original.csv        the 66 competition training photos
  train_*.csv               earlier training-file versions
  test.csv                  36 test photos
  images/                   all images (competition + external)
features/                   cached embeddings and auxiliary scores (*.npz)
features_backup_before_external/   features used by the 0.94155 submission
features_backup_before_openbags/   features before the open-bag photos were added
best_0.94155/               frozen code + data that reproduce the best public file
logs/                       CV tables (*.csv), run logs (*.log), error sheets (*.jpg)
chore_classifier.ipynb      CLIP baseline and data exploration
dev_predictions.ipynb       early dev split (leaky, kept for reference)
model_comparison*.ipynb     model / view-mode comparison, 4 training-data variants
per_chore.ipynb             per-chore binary models + nested selection check
extract_features.py, train.py, _clf.py     backbone features and model grid
overflow_aux.py, bag_tiles.py, aux_cv.py, make_submission_aux.py   auxiliary scores (best CV model)
*_aug.py, ens_*.py          augmentation and ensembling experiments
dish*.py, soiled_windows.py, relabel_sink_cv.py, drop_dishes_cv.py   dishes experiments
ft_dinov2.py                DINOv2 fine-tuning
submission_*.csv            prediction files (section 4)
```

## 7. Software and licences

- Python with numpy, pandas, scipy, scikit-learn 1.9.1, matplotlib, Pillow (versions in
  `best_0.94155/README.md`). Virtual environment: `../.venv`.
- PyTorch, `timm` (DINOv2), `open_clip_torch` (CLIP, EVA02, SigLIP). Fine-tuning ran on an M1 Pro (16 GB,
  MPS), about 6–8 min per fold.
- All random seeds are fixed; `make_submission_aux.py` gives identical files on rerun.

Models: DINOv2 (Apache-2.0), OpenCLIP (MIT) with OpenAI CLIP / LAION / EVA02 / SigLIP weights
(*TODO: list each weight licence*). Data: competition photos (Unsplash), external images listed with licences
in `../external_images/CREDITS.csv`, CDCM (CC BY 4.0, Rodrigo Laguna, Kaggle `rodrigolaguna/clean-dirty-containers-in-montevideo`). Sources without a stated
licence were not used.
