"""Error review: which images does a model get wrong, and what do they look like?

WHAT THIS FILE DOES
    baseline.py saved, for every model it tested, the OUT-OF-FOLD (OOF) probabilities: for each of the 941
    labelled images (train + validation), P(artifact) predicted by a model that never saw that image during
    training (averaged over the 3 CV repeats). This script takes one of those files and:
      1. labels every image as correct, FN or FP (see below)
      2. saves a table of all 941 images with their probability, label, prediction and a few quality numbers
      3. makes two contact sheets (grids of thumbnails) of the model's most confident mistakes, so they can be
         looked at by eye
      4. prints how many mistakes of each kind there are, and the average sharpness / brightness of each group
    Using OOF probabilities means every labelled image can be reviewed (941), not only the 118 validation ones.

THE TWO KINDS OF MISTAKE (positive class = "artifact present" = 1)
    FN  false negative = a MISS:         the image HAS an artifact (label 1) but the model said clean (0)
    FP  false positive = a FALSE ALARM:  the image is CLEAN (label 0) but the model said artifact (1)
    A prediction is 1 when the probability is >= 0.5, the same threshold as in baseline.py.

WHAT IT FOUND FOR RUN1 (all3+quality_svm_rbf)
    61 misses and 50 false alarms out of 941 (on the 118 validation images: 9 and 7).
    Most confident mistakes on both sheets: fog / mist landscapes and raindrops on windows, labelled both ways.
    (A separate near-duplicate check, logs/near_duplicates.csv and logs/conflicting_duplicates.jpg, then found
    three identical image pairs with opposite labels. That check was run on its own, not by this script.)

OUTPUTS
    logs/errors_<name>.csv       one row per image: row, image_id, filename, artifact_present, split, prob,
                                 g_lap_var, g_bright_mean, g_clip_hi, g_clip_lo, g_dark_channel_mean, pred, err
    logs/errors_<name>_FN.jpg    the 36 most confident misses       (lowest probability first)
    logs/errors_<name>_FP.jpg    the 36 most confident false alarms (highest probability first)
    Each thumbnail is captioned  "#<row> p=<probability> <split>",  e.g. "#542 p=0.02 tra".
    #<row> is the ROW NUMBER in errors_<name>.csv (0-822 = train.csv rows, 823-940 = validation.csv rows),
    NOT the number in the image's filename; look the row up in the CSV to find the file.

USAGE
    python review_mistakes.py [oof name]     default: all3+quality_svm_rbf (run1)
    The name is the part between "oof_" and ".npy" in features/, e.g. eva02_l+quality_mlp_strong.
    Rerunning gives identical files.
"""
import sys                                   # command-line argument
from pathlib import Path                     # file paths

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps   # opening images, drawing captions, EXIF rotation

from baseline import load_labels             # the train and validation tables

ROOT = Path(__file__).parent

# ----------------------------------------------------------------------------------------------------------
# 1. Build the table: one row per labelled image
# ----------------------------------------------------------------------------------------------------------
# Which model to review: the command-line argument, or run1 by default.
name = sys.argv[1] if len(sys.argv) > 1 else "all3+quality_svm_rbf"
tr, va = load_labels()
# Train rows first, then validation rows: the SAME order baseline.py used when saving the OOF probabilities,
# so row i of this table matches value i of the .npy file. ignore_index=True numbers the rows 0..940.
df = pd.concat([tr.assign(split="train"), va.assign(split="validation")], ignore_index=True)
df["prob"] = np.load(ROOT / f"features/oof_{name}.npy")      # P(artifact) for each image, 941 values

# Add five of the quality numbers (from quality_features.py) to help describe the mistakes:
#   g_lap_var (sharpness), g_bright_mean (brightness), g_clip_hi / g_clip_lo (share of pure white / black pixels),
#   g_dark_channel_mean (haze).
q = np.load(ROOT / "features/quality.npz", allow_pickle=True)
qd = pd.DataFrame(q["X"], columns=q["names"]).assign(image_id=q["ids"])   # quality table with an image_id column
# merge = join the two tables on image_id (matching by id, not by position). how="left" keeps every row of df.
df = df.merge(qd[["image_id", "g_lap_var", "g_bright_mean", "g_clip_hi", "g_clip_lo", "g_dark_channel_mean"]],
              on="image_id", how="left")

# Prediction: 1 if the probability is at least 0.5, else 0.
df["pred"] = (df.prob >= 0.5).astype(int)
# Type of result for each image:
#   prediction equals the label          -> ""  (correct)
#   otherwise, prediction is 1 (label 0) -> "FP" (false alarm)
#   otherwise, prediction is 0 (label 1) -> "FN" (miss)
# np.where(condition, a, b) picks a where the condition is true and b elsewhere; nested once for the 3 cases.
df["err"] = np.where(df.pred == df.artifact_present, "", np.where(df.pred == 1, "FP", "FN"))
# Save the full table. index_label="row" names the row-number column (the "#<row>" on the contact sheets).
df.to_csv(ROOT / f"logs/errors_{name}.csv", index_label="row")
# Counts, e.g. {'': 830, 'FN': 61, 'FP': 50} | validation: {'': 102, 'FN': 9, 'FP': 7}
print(df.err.value_counts().to_dict(), "| validation:", df[df.split == "validation"].err.value_counts().to_dict())


# ----------------------------------------------------------------------------------------------------------
# 2. Contact sheets
# ----------------------------------------------------------------------------------------------------------
def sheet(rows, path, cols=6, cell=230):
    """Paste the images in `rows` into one picture: a grid `cols` wide, each cell `cell` x `cell` px plus a
    20 px strip for the caption. Saves it as a JPEG at `path`."""
    n = len(rows)
    # Blank white canvas. Number of grid rows = n / cols rounded UP ((n + cols - 1) // cols does that).
    im = Image.new("RGB", (cols * cell, ((n + cols - 1) // cols) * (cell + 20)), "white")
    d = ImageDraw.Draw(im)   # a "pen" for writing the captions onto the canvas
    # k = position on the sheet (0, 1, 2, ...); i = the image's row number in the table; r = that row's data.
    for k, (i, r) in enumerate(rows.iterrows()):
        # Open the image upright and in RGB (same as everywhere else), shrink it to fit the cell.
        t = ImageOps.exif_transpose(Image.open(ROOT / "data" / r.filename)).convert("RGB")
        t.thumbnail((cell - 10, cell - 10))
        # Top-left corner of cell k: column k % cols, grid row k // cols; + 5 px margin.
        x, y = (k % cols) * cell + 5, (k // cols) * (cell + 20) + 5
        im.paste(t, (x, y))
        # Caption under the thumbnail, e.g. "#542 p=0.02 tra" (row number, probability, first 3 letters of split).
        d.text((x, y + cell - 8), f"#{i} p={r.prob:.2f} {r.split[:3]}", fill="black")
    im.save(path, quality=88)   # JPEG quality 88: small file, still clear


# The most CONFIDENT mistakes are the most informative: the model was sure and still wrong.
#   Misses (FN): sorted by probability, LOWEST first (the model was most sure these were clean).
#   False alarms (FP): sorted by probability, HIGHEST first (most sure these were artifacts).
# head(36) keeps 36 of each = a 6 x 6 grid.
fn = df[df.err == "FN"].sort_values("prob").head(36)
fp = df[df.err == "FP"].sort_values("prob", ascending=False).head(36)
sheet(fn, ROOT / f"logs/errors_{name}_FN.jpg")
sheet(fp, ROOT / f"logs/errors_{name}_FP.jpg")

# ----------------------------------------------------------------------------------------------------------
# 3. Do the mistakes differ in sharpness / brightness from the correct ones?
# ----------------------------------------------------------------------------------------------------------
# Average of four quality numbers for misses, false alarms and correct images. For run1 the differences were
# small (e.g. sharpness 4.96 / 4.21 / 5.16): the mistakes are not simply the blurriest or darkest images.
for e in ("FN", "FP", ""):
    s = df[df.err == e]
    print(f"{e or 'correct':8s} n={len(s):4d}  lap_var {s.g_lap_var.mean():.2f}  bright {s.g_bright_mean.mean():.2f}  "
          f"clip_hi {s.g_clip_hi.mean():.3f}  clip_lo {s.g_clip_lo.mean():.3f}")
