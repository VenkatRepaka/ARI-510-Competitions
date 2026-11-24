# Submission summary: Chore completion classifier

Short summary in five parts: task, approaches, runs, what worked, code. "Dev F1" is always the macro F1 from 10 × repeated stratified 5-fold
cross-validation over the 66 labelled training photos (see Approaches, "Dev set").

---

## 1. Task

Kaggle: *Chore completion classifier*.

Each photo shows one of two household chores, Washing Dishes or Taking Out Trash, and the chore is either Complete or
Incomplete. The model must give each photo one of these 4 classes. The score is macro F1 (the F1 of each class,
averaged). The data is small: 66 labelled training photos and 36 test photos. The chore type is easy to see; the hard
part is Complete against Incomplete, which often depends on a small detail, such as one bowl left next to a clean sink,
one untied bag, or trash spilling over the edge of a bin.

---

## 2. Approaches

**Plan.** With only 66 photos, training an image network from scratch is not possible. So the plan was: use frozen
pretrained image models to turn each photo into features, train small classifiers on them, and then find out what is
missing by looking at the photos that are wrong. The errors showed two gaps: some kinds of scene appear only once in
the training data, and small details are lost in a whole-photo feature. Licensed extra photos and two small helper
classifiers on zoomed-in parts of the photo were added to close these gaps.

**Preprocessing.** Each photo is opened with Pillow, turned by its EXIF orientation, converted to RGB and reduced to at
most 1536 px on the long side. Each image model then uses its own resize and normalisation. Nothing else is cleaned.
Photo ids and file names are never used as features. For each photo, 7 views are encoded: the photo, its mirror image,
and 5 crops (four corners and the centre, 2/3 of the width and height). The "full" view used by the models is the
average of the photo and its mirror image. Features are L2-normalised. For DINOv2 the feature is the class token next
to the mean patch token (2048 numbers).

**Dev set and evaluation.** 66 photos are too few for one fixed dev split (one photo changes macro F1 by about 0.03).
So every model is scored with repeated stratified 5-fold cross-validation: 5 folds × 10 repeats (seed 0), and the mean
macro F1 over the 10 repeats is the dev score. A photo and everything made from it (mirror, crops, tiles, augmented
views) always stay on the same side of a split. Extra photos are only added to the training side, and only the 66
original photos are scored, so all dev scores can be compared. The spread between repeats is about ±0.02, so smaller
differences were treated as noise. For each model, the number of repeats in which each photo is wrong (0–10) was also
recorded; this shows which photos a change fixes or breaks. Models were chosen by the dev score, not by the public
leaderboard. The test photos were never looked at or labelled.

**Models tried.**

- IMAGE MODEL × CLASSIFIER (66 photos only). Five frozen image models: DINOv2 ViT-L/14, SigLIP SO400M, EVA-02-CLIP
ViT-L/14, OpenCLIP ViT-H/14 (LAION-2B) and OpenAI CLIP ViT-L/14, alone and all five joined. Four classifiers on
each: logistic regression (C = 1), RBF SVM (C = 3), extra trees (300 trees) and a neural network (below). Results:
DINOv2 + RBF SVM 0.800, DINOv2 + extra trees 0.819, all five joined + extra trees 0.826, OpenAI CLIP about 0.70.
DINOv2 was kept, because its features also work well on small parts of a photo.
- NEURAL NETWORK (MLP), trained in PyTorch on the frozen features. Code: class `TorchMLP` in `common.py` (line 156),
used as the classifier "neural net (MLP)" in notebooks 02, 05 and 06. Network: dropout 0.2 → Linear(input, 256) →
GELU → dropout 0.5 → Linear(256, 4); class-weighted cross-entropy with label smoothing 0.1, AdamW, cosine
learning-rate schedule, a fixed 600 steps of batch 32 (no early stopping), seed 0. It was scored next to the other
classifiers at every step. On the 66 photos (notebook 02): DINOv2 0.776, SigLIP 0.808 (the best result for SigLIP),
all five joined 0.795. On the final training set and features (notebooks 05 and 06): 0.847, against 0.912 for the
RBF SVM; the 6 helper numbers next to 2048 image numbers helped the SVM and logistic regression but not the MLP.
Notebook 06 trains it on all 105 photos and writes `submission_mlp.csv` (agrees with run6 on 33 of 36 test photos).
Not submitted.
- MULTI-CROP AVERAGING: averaging the photo, mirror and 5 crops instead of photo and mirror only. No gain.
- EXTRA TREES ON 5 IMAGE MODELS: one extra-trees model per image model, class probabilities averaged. Dev 0.821.
Submitted as run 2, and again as run 7 with a description.
- AUGMENTATION: extra training views of each training photo: 20 random crops, colour changes, zoom, 60 % crops turned
in different directions, colour swaps (red ↔ blue, green ↔ yellow). The views of a photo are only used while that
photo is on the training side (an early version that augmented before splitting scored a false 0.995). RBF SVM 0.800
→ 0.81–0.83; logistic regression with 20 augmented views per photo was submitted as run 1. A check showed that crops
of Incomplete photos do not lose their label more often than the full photos, but after extra photos were added,
augmentation gave no further gain.
- EXTRA PHOTOS: photos with a clear licence (Unsplash, Pexels, Pixabay, Creative Commons) and 4 own photos, for the
scene types that appear only once in the training data. Rules: (1) a licence that can be checked, so Pinterest saves
were not used; (2) not a copy of a training or test photo: each new photo was compared with all 102 original photos
(DINOv2 cosine ≥ 0.90 = same picture), which found 13 copies, three of them test photos, and all were dropped;
(3) the label follows the photo, not the score. Groups were added one at a time (RBF SVM, with the helpers below):
66 photos 0.842; + 19 bag, bin and sink photos 0.895; + 7 dishes photos 0.905; + 4 closed wheelie bins 0.912;
  + 4 own photos 0.905 (fixed #2, broke the clean sink #12); + 5 clean sinks 0.912 (fixed #12 again). Final set:
  39 extra photos. Earlier versions with 15 and 12 web photos were submitted as runs 3 and 4.
- OVERFLOW HELPER: each photo is cut into 13 overlapping tiles (2 × 2 and 3 × 3 grids). A logistic regression trained
on the CDCM dataset (smartphone photos of clean and dirty street containers, CC BY 4.0) scores each tile for "an
overflowing bin". The highest tile score and the mean of the two highest, for DINOv2 and SigLIP features, are added
to the photo's features (4 numbers). Accuracy on held-out CDCM photos: 0.895 (DINOv2) and 0.935 (SigLIP). The helper
never sees the labels of the original photos. Dev F1 0.882 → 0.905.
- BAG HELPER: a logistic regression trained on the 12 licensed bag photos (7 tied, 5 open), each as photo, mirror and
13 tiles, scores each tile for "an open bag". It adds 2 numbers (highest and mean of the two highest). Dev F1
0.905 → 0.912. An earlier rule-based version (tile similarity to tied and open bag photos) was used in run 5 and was
later replaced by this learned version.
- FINAL MODEL: DINOv2 full view (2048) + 4 overflow numbers + 2 bag numbers → RBF SVM (C = 3, gamma "scale",
class-balanced), trained on 66 + 39 photos. Dev F1 0.912 ± 0.013. Submitted as run 6.

**Hyperparameter selection.** The classifier settings were fixed before the results were seen, using standard values:
logistic regression C = 1, RBF SVM C = 3 with gamma "scale", extra trees 300 trees with √(features) per split, MLP as
described above; all class-balanced. With 66 photos, a grid search would mostly fit the noise, so the comparison was
between feature sets, training data and model types, not between settings. The tile sizes (2 × 2 and 3 × 3 grids) and
the tile summaries (highest score and mean of the two highest) were also fixed in advance. All random seeds are fixed.

**Error review.** After each step, the photos wrong in most repeats were viewed in a contact sheet. This led to the
extra photo groups (closed bins for #63, a dirty bowl next to a clean sink for #2, clean sinks for #12) and to the two
helpers (overflowing bins #99 and #100, open bag #13: wrong 10/10 → 1/10). Five photos are still wrong in every repeat: #26 (sorted
recycling, labelled Complete), #31 and #60 (clean-looking dishes and a full drying rack, labelled Incomplete), #77
(bags) and #101 (bowls left out, labelled Complete). In these photos the label depends on a judgement that the same
kind of scene does not always get in the training data.

**External data and models.** Frozen pretrained models: DINOv2 (Apache-2.0, through `timm`), SigLIP, EVA-02-CLIP,
OpenCLIP ViT-H/14 and OpenAI CLIP (through `open_clip_torch`, MIT). External data: 39 licensed photos (Unsplash,
Pexels, Pixabay, CC BY / BY-SA / BY-NC-SA, 4 own photos; every file with source and licence in `data/external.csv`)
and the CDCM dataset (Rodrigo Laguna, Kaggle, CC BY 4.0). No large language model was used as an approach. An AI
coding assistant (Claude) helped to write the code and run the experiments.

---

## 3. Runs

Kaggle baselines (public): trained baseline 0.747, random 0.326, majority 0.125.

run1 | DINOv2 features, logistic regression on 20 augmented views per photo, 66 photos | — | 0.883  
run2 | extra trees on 5 frozen image models (photo + mirror), probabilities averaged, 66 photos | 0.821 | 0.94155  
run3 | DINOv2 features, RBF SVM C=3, 66 photos + 15 web photos | 0.845 | 0.88311  
run4 | DINOv2 + overflow tile helper (CDCM), RBF SVM C=3, 66 + 12 web photos | 0.890 | 0.88311  
run5 | run4 + rule-based open-bag score, 66 + 19 web photos; dishes photos by a separate logistic regression | 0.916 | 0.88311  
run6 | DINOv2 + overflow tile helper + learned bag helper, RBF SVM C=3, 66 + 39 licensed photos | 0.912 | 0.88311  
run7 | extra trees on 5 frozen image models (photo + mirror), probabilities averaged, 66 photos | 0.821 | 0.94155

Also prepared, not submitted: the neural network (PyTorch MLP) on the run6 input and photos, dev F1 0.847,
file `submission_mlp.csv`.

Run 1 was scored before the 10-repeat CV was set up, so it has no comparable dev score. The dev F1 of run 2 is the
10-repeat CV from notebook 07; a shorter CV at the time gave 0.812.

---

## 4. What worked

The best dev score came from more and better training data, not from a bigger model. Adding licensed photos of the
scene types that were missing raised the dev F1 from 0.842 to 0.912, while changing the classifier moved it by
about 0.02 and the best image models were within 0.02 of each other. With 66 photos, each new kind of scene (a closed wheelie bin, a single dirty bowl next to a
clean sink, an empty clean sink) gives the classifier an example it did not have before. The copy check was very
important here: many stock photos were the same pictures as the training or test photos, and using them would have
made the dev score look better than it is.

The second gain came from looking at parts of the photo. A whole-photo feature describes the whole scene, so a small
detail that decides the label (trash over the edge of one bin, one open bag in a street) hardly changes it. Two small
helpers that score 13 zoomed-in tiles of each photo raised the trash classes, and the dev F1 rose from 0.882 to 0.912.
They are trained on separate data (the CDCM containers and 12 bag photos), so they add knowledge that the 66 photos do
not contain.

The neural network on the frozen features was close to the other classifiers on the 66 photos (0.78–0.81) but
fell behind on the final input (0.847 against 0.912). Augmentation did not
help once the extra photos were added, because changed copies of the same photos do not show new kinds of scene.

The public leaderboard does not agree with the dev score: run 2 (dev 0.821) has the best public score (0.94155) and
runs 3–5 (dev 0.845–0.916) all have 0.88311. The public part has only about 18 photos, so 0.94155 means about one
wrong photo and 0.88311 about two. One photo is within the noise, so the dev score is the better guide. For the final
leaderboard, the run with the best dev score is the one the results support.

---

## 5. Code

https://github.com/VenkatRepaka/ARI-510-Competitions/tree/master/chore-completion-classifier

Setup: `extract_features.py` (image features, once), then the notebooks in order. Details in `README.md`.

- run6: notebook `06_final_model.ipynb` → `submission_final.csv`.
- run2 and run7: notebook `07_best_public_submission.ipynb` → `submission_best_public.csv` (the same 36 predictions as
the run2 file `best_0.94155/submission_public0.94155_extratrees.csv`). The original run2 code is kept in
`best_0.94155/`.
- neural network (not submitted): notebook `06_final_model.ipynb` → `submission_mlp.csv`.
- run1, run3, run4, run5: earlier versions of the code; the submitted files are kept in `submissions/`.

All random seeds are fixed (CV splits seed 0, extra trees seed 0, PyTorch seed 0), so running the code again gives
the same files.
