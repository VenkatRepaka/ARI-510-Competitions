# Chore completion classifier

A personal project on the Kaggle competition *Chore completion classifier*.

Each photo shows one of two household chores, **Washing Dishes** or **Taking Out Trash**. The chore is either
**Complete** or **Incomplete**. So there are 4 classes, and the score is **macro F1**: the F1 score of each class,
averaged over the 4 classes. There are 66 labelled training photos and 36 test photos without labels.

**Final model (dev macro F1 0.912 ± 0.013).** A pretrained image model (DINOv2) turns each photo into a list of
numbers. Two small helper classifiers look at zoomed-in parts of the photo and add 6 more numbers: one looks for
overflowing bins, the other for open bags. An RBF support vector machine (SVM) makes the final decision. It is trained
on the 66 original photos and 39 extra photos with a free licence.

## Contents

1. [Folder layout](#folder-layout)
2. [Setup](#setup)
3. [How to run](#how-to-run)
4. [How the models are tested](#how-the-models-are-tested)
5. [Method, step by step](#method-step-by-step)
6. [Results](#results)
7. [Kaggle submissions](#kaggle-submissions)
8. [Software, data and licences](#software-data-and-licences)

## Folder layout

| file or folder | what it contains |
|---|---|
| `extract_features.py` | turns every photo into image features with five pretrained image models; saves them in `features/` |
| `common.py` | code that all notebooks share: loading data and features, the four classifiers, cross-validation, the copy check |
| `01_data_and_features.ipynb` | the data, the test method, and the five image models |
| `02_model_comparison.ipynb` | 5 image models × 4 classifiers on the 66 original photos |
| `03_augmentation.ipynb` | extra training views made from each photo (crops, colour, zoom, rotation) |
| `04_external_photos.ipynb` | rules for extra photos, and what each group of extra photos adds |
| `05_helpers.ipynb` | the two helper classifiers (overflowing bins, open bags): how they are built and what they add |
| `06_final_model.ipynb` | the final model (run 6): test score, remaining errors, `submission_final.csv`; also the neural network file `submission_mlp.csv` |
| `07_best_public_submission.ipynb` | rebuilds run 2, the submission with the best public score (0.94155); run 7 uses its file |
| `submission_final.csv` | predictions of the final model for the 36 test photos (run 6) |
| `submission_mlp.csv` | predictions of the neural network on the same input (not submitted) |
| `submission_best_public.csv` | predictions of run 2, rebuilt by notebook 07 (run 7) |
| `submission.md` | a short summary of the project: task, approaches, runs, what worked, code |
| `data/train_original.csv`, `data/test.csv` | ids and labels of the 66 training photos; ids of the 36 test photos |
| `data/external.csv` | every extra photo that was tried: group, label, source, licence, and whether it is used |
| `data/images/` | all photos (`image_<id>.jpg`; extra photos are `image_<id>_external.jpg`) |
| `data/images/external/` | copies of the 39 extra photos in the final model, sorted by label and scene, with `CREDITS.csv` |
| `features/` | saved image features and helper scores (about 0.5 GB, not in git) |
| `best_0.94155/` | the original code and file of run 2, kept unchanged |
| `submissions/` | earlier submission files |
| `logs/` | log files and contact sheets from earlier experiments |

## Setup

The code uses Python 3 with a virtual environment one folder up (`../.venv`). It needs these packages:

```bash
../.venv/bin/pip install numpy pandas scipy scikit-learn torch torchvision pillow timm open_clip_torch matplotlib jupyter nbconvert
```

The image models are downloaded automatically the first time `extract_features.py` runs (about 6 GB in total).
The code runs on a CPU, an Apple GPU (MPS) or an NVIDIA GPU.

## How to run

All commands start in this folder:

```bash
cd chore-completion-classifier
```

**Step 1: image features.** This step encodes every photo with the five image models. It only encodes photos that are
not yet in `features/`, so it is fast when the folder is already filled. On an empty folder it takes about
30–60 minutes.

```bash
../.venv/bin/python extract_features.py
```

**Step 2: the notebooks, in order.** They can be opened in Jupyter, or run from the command line:

```bash
for f in 0*.ipynb; do ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$f"; done
```

| notebook | time (features saved) | writes |
|---|---|---|
| 01 | under 1 min | — |
| 02 | about 10 min | — |
| 03 | about 2 min | — |
| 04 | about 2 min | — |
| 05 | about 15 min the first time, then 1 min | `features/overflow_tiles.npz`, `features/bag_learned.npz` (helper scores) |
| 06 | about 1 min | `submission_final.csv`, `submission_mlp.csv` |
| 07 | about 1 min | `submission_best_public.csv` |

Notebooks 04 and 06 need the helper scores from notebook 05. These files are already in `features/`, so the notebooks
can also run on their own. All random seeds are fixed, so a second run gives the same numbers and the same
submission files.

**Which run comes from which code.**

| run | how to make the file |
|---|---|
| runs 2 and 7 (best public score) | notebook 07 → `submission_best_public.csv` (the same predictions as `best_0.94155/submission_public0.94155_extratrees.csv`) |
| run 6 (final model) | notebook 06 → `submission_final.csv` |
| runs 1, 3, 4, 5 | earlier versions of the code; the submitted files are kept in `submissions/` |

## How the models are tested

There is no separate dev set, because 66 photos are too few to split once. Every result in this project uses the
same method:

- **10 × repeated stratified 5-fold cross-validation (CV).** The 66 original photos are split into 5 parts with the
  same class mix. Each part is predicted once by a model trained on the other 4 parts. This is repeated 10 times with
  different splits (seed 0), and the mean macro F1 is reported.
- **No leaks between training and testing.** A photo and everything made from it (mirror, crops, tiles) are always on
  the same side of a split.
- **Extra photos only train.** Extra photos are added to the training side of every split. Only the 66 original photos
  are scored, so every score can be compared with every other score.
- **Noise level.** One photo more or less wrong changes macro F1 by about 0.03. The spread between repeats is about
  ±0.02. Differences under 0.02 are treated as noise.
- **Photos wrong in every repeat.** For each model, the notebooks count how often each photo is wrong (0–10 times).
  This shows which photos a change fixes or breaks.
- The test photos were never looked at or labelled. They are only used for the final predictions. The public Kaggle
  score covers only part of the test photos, so it was not used to choose models.

## Method, step by step

### 1. Image features (notebook 01, `extract_features.py`)

A pretrained image model turns each photo into a fixed list of numbers (an *embedding*). The classifiers learn only
from these lists. Five frozen models (no retraining) were compared:

| short name | model | trained on |
|---|---|---|
| `dinov2_l` | DINOv2 ViT-L/14 (Oquab et al., 2023) | 142 million images, without labels |
| `siglip_so400m` | SigLIP SO400M (Zhai et al., 2023) | image–text pairs (WebLI) |
| `eva02_l` | EVA-02-CLIP ViT-L/14 | image–text pairs |
| `vith14_laion` | OpenCLIP ViT-H/14 | image–text pairs (LAION-2B) |
| `vitl14_openai` | CLIP ViT-L/14 (Radford et al., 2021) | image–text pairs (OpenAI) |

For each photo, `extract_features.py` encodes 7 views: the photo, its mirror image, and 5 crops (four corners and the
centre, each 2/3 of the photo's width and height). For DINOv2 the features are the class token and the mean of the
patch tokens (2 × 1024 = 2048 numbers). The **"full" view**, used almost everywhere, is the average of the photo and
its mirror image.

### 2. Image model and classifier (notebook 02)

Each image model, and all five joined together, is tried with four classifiers. All of them use class-balanced
weights:

- **logistic regression** (C = 1),
- **RBF SVM**: a support vector machine with a radial basis function kernel (C = 3, gamma "scale"),
- **extra trees**: 300 random decision trees, √(number of features) per split,
- **neural network (MLP)** in PyTorch (`common.TorchMLP`): dropout 0.2 → 256 hidden units with GELU → dropout 0.5 →
  4 outputs; AdamW, cosine learning rate, label smoothing 0.1, 600 steps.

With only the 66 photos, DINOv2 with an RBF SVM scores 0.800, extra trees 0.819, logistic regression 0.808 and the
neural network 0.776. All five models joined with extra trees score 0.826, and OpenAI CLIP is the weakest (about
0.70). DINOv2 is used from here on, because its features also work well on small parts of a photo (step 5).
The settings above were fixed in advance and not tuned, because 66 photos are too few to tune many settings fairly.

### 3. Data augmentation (notebook 03)

Extra training rows are made from each training photo: crops, mirror, colour changes, zoom, rotation and colour swaps.
The changed views of a photo are only used while that photo is on the training side. The gain is small (SVM 0.800 →
0.81–0.83, close to the noise level), and it disappears once the extra photos of step 4 are added. So augmentation
is not used in the final model.

### 4. Extra photos (notebook 04)

Some kinds of scene appear only once in the 66 photos, for example a closed wheelie bin, or one dirty bowl next to a
clean sink. Adding real photos of these scenes helped more than anything else. Each extra photo must follow three
rules:

1. **A clear licence** (Unsplash, Pexels, Pixabay, Creative Commons) or an own photo. Photos without a licence that can
   be checked (for example Pinterest saves) are only used for comparison, never in the final model.
2. **Not a copy of an original photo.** Many stock photos turned out to be the same pictures as the original training
   or test photos. Each new photo is compared with all 102 original photos (DINOv2 cosine similarity ≥ 0.90 = same
   photo) and dropped if it matches. This check found 13 copies, three of them test photos.
3. **The label follows the photo**, not the score. Photo 913 shows a mug in the sink, so it is *Incomplete*, even
   though *Complete* scores higher.

The groups were added one at a time (RBF SVM with the helpers of step 5):

| training photos | dev macro F1 |
|---|---|
| 66 original photos | 0.842 |
| + 19 bag, bin and sink photos | 0.895 |
| + 7 dishes photos | 0.905 |
| + 4 closed wheelie bins (fixed #63) | 0.912 |
| + 4 own photos (fixed #2, but broke the clean sink #12) | 0.905 |
| + 5 clean sinks (fixed #12 again) = **final set, 39 extra photos** | **0.912** |

### 5. Two helper classifiers on zoomed-in tiles (notebook 05)

The whole-photo embedding describes the whole scene, so a small detail (trash spilling over one bin, one untied bag
in a street) hardly changes it. Each photo is therefore cut into 13 overlapping tiles (a 2 × 2 and a 3 × 3 grid), and
two small logistic regression models score every tile:

| helper | trained on | question | numbers added to each photo |
|---|---|---|---|
| overflow | CDCM dataset: smartphone photos of clean and dirty street containers | "Is this part an overflowing bin?" | 4: highest tile score and mean of the two highest, for DINOv2 and SigLIP |
| bag | the 12 licensed bag photos (7 tied, 5 open), each as photo, mirror and 13 tiles | "Is this part an open bag?" | 2: highest tile score and mean of the two highest |

The helpers never see the labels of the original photos, so their scores can be computed once without leaking
answers. Only the final classifier is retrained in each split. With the final training set, DINOv2 alone scores
0.882, the overflow helper raises this to 0.905, and the bag helper to 0.912.

### 6. Final model (notebook 06)

- **Input:** 2048 DINOv2 numbers (full view) + 4 overflow numbers + 2 bag numbers = 2054 numbers per photo.
- **Classifier:** RBF SVM, C = 3, gamma "scale", class-balanced.
- **Training data:** 66 original photos + 39 extra photos = 105 photos.
- **Dev macro F1:** 0.912 ± 0.013 (Washing Dishes 0.893, Taking Out Trash 0.931).
- The other classifiers on the same input: logistic regression 0.895, extra trees 0.863, neural network 0.847.
- The model is trained once on all 105 photos and predicts the 36 test photos → `submission_final.csv`.

### 7. The best public submission (notebook 07)

Run 2 uses extra trees on each of the five image models (full view, 66 original photos only) and averages the five
models' class probabilities. Its dev macro F1 is only 0.821, but it got the best public score (0.94155). The public
score covers only about half of the 36 test photos, so one photo changes it by several points. 0.94155 means about one
wrong photo, and 0.88311 means about two. The public score therefore says little about which model is better.
Notebook 07 rebuilds this run from `common.py` and checks that it gives the same 36 predictions as the file that was
submitted.

## Results

All numbers are dev macro F1 from the CV described above (RBF SVM unless noted).

| step | dev macro F1 | notebook |
|---|---|---|
| Kaggle baseline (public score of the trained baseline) | 0.747 | — |
| DINOv2, 66 photos | 0.800 (extra trees 0.819, all 5 models joined 0.826) | 02 |
| + augmentation | 0.81–0.83 | 03 |
| + helpers, 66 photos | 0.842 | 04 |
| + 19 licensed bag, bin and sink photos | 0.895 | 04 |
| + 7 dishes photos, 4 closed bins, 4 own photos, 5 clean sinks (final set) | **0.912** | 04, 06 |
| final set, DINOv2 only (no helpers) | 0.882 | 05 |
| run 2: extra trees, 5 image models, 66 photos | 0.821 | 07 |

**Photos the final model gets wrong in every repeat:** #26 (sorted recycling, labelled Complete), #31 and #60
(clean-looking dishes and a full drying rack, labelled Incomplete), #77 (bags), #101 (bowls left out, labelled
Complete). These labels depend on small details, or the same kind of scene has different labels in different photos.

## Kaggle submissions

Runs 1–5 were submitted without a description on Kaggle; they are matched by submission order and by file. Runs 6
and 7 have a run ID and a one-line summary in their Kaggle description (see `submission.md`).

| run | file | model | dev F1 | public F1 |
|---|---|---|---|---|
| Kaggle baseline | — | trained baseline / random | — | 0.747 / 0.326 |
| run 1 | `submissions/submission_aug_dinov2.csv` | DINOv2, logistic regression on 20 augmented views per photo | — | 0.883 |
| run 2 | `best_0.94155/submission_public0.94155_extratrees.csv` | extra trees on 5 image models, 66 photos | 0.821 | **0.94155** |
| run 3 | `submissions/submission_dinov2_svm_external.csv` | DINOv2, RBF SVM, + 15 web photos | 0.845 | 0.88311 |
| run 4 | `submissions/submission_dinov2_tiles_svm.csv` | DINOv2 + overflow helper, RBF SVM, + 12 web photos | 0.890 | 0.88311 |
| run 5 | `submissions/submission_dinov2_open_routed.csv` | run 4 + rule-based open-bag score + 19 web photos; dishes photos sent to a separate logistic regression | 0.916 | 0.88311 |
| run 6 | `submission_final.csv` | final model of this README | 0.912 | to add |
| run 7 | `submission_best_public.csv` | run 2 again, with a description (same predictions) | 0.821 | to add |
| — | `submission_mlp.csv` | neural network (MLP) on the run 6 input, not submitted | 0.847 | — |

**How each run was made.**

- **Run 6:** notebook 06 (`submission_final.csv`). The MLP file `submission_mlp.csv` comes from the same notebook.
- **Runs 2 and 7:** notebook 07 (`submission_best_public.csv`). The frozen original code is in `best_0.94155/`.
- **Runs 1, 3, 4 and 5:** earlier versions of the code; the submitted files are kept in `submissions/`.
- The dev F1 of run 2 is the 10-repeat CV of notebook 07 (0.821). At the time of submission, a shorter CV gave 0.812.

## Software, data and licences

- **Software:** Python with numpy, pandas, scipy, scikit-learn, PyTorch, Pillow and matplotlib; `timm` (DINOv2,
  Apache-2.0) and `open_clip_torch` (MIT) for the image models.
- **Original photos:** from the Kaggle task (Unsplash and Pexels photos).
- **Extra photos:** Unsplash, Pexels, Pixabay and Creative Commons photos (CC BY 2.0, CC BY-SA 2.0–4.0,
  CC BY-NC-SA 2.0), and 4 own photos. Every file is listed with source and licence in `data/external.csv` and
  `data/images/external/CREDITS.csv`. Photo 944 is AI-generated (Pixabay).
- **CDCM dataset** (Rodrigo Laguna, Kaggle `rodrigolaguna/clean-dirty-containers-in-montevideo`, CC BY 4.0): trains
  the overflow helper.
- Photos without a licence that can be checked (Pinterest saves) are not used in the final model.
