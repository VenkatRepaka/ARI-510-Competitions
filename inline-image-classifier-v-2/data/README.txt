IMAGE ARTIFACT CLASSIFICATION

Predict artifact_present: 0 = none of the five listed artifacts; 1 = at least one.
Categories: physical obstruction, exposure fault, lens contamination, optical/motion blur, and lens fog/condensation.

train.csv and validation.csv contain image_id, filename, artifact_present.
filename is a path relative to the root of this package.
test.csv contains only image_id and filename, with no test labels.
sample_submission.csv shows the required prediction format: image_id,artifact_present.
Submit exactly 0 or 1 for each test image, not a probability. The primary leaderboard metric is binary F1 for artifact present (1).

Images are in images/train, images/validation, and images/test.
Use training for fitting and validation for development. Do not manually label test images.
Treat image_id and filenames as identifiers, not model features.
HEIC/AVIF images were converted to full-resolution PNG; other images are unchanged.
Load images as RGB and honor EXIF orientation; do resizing/normalization for your model.
Only the primary image of a multi-image photo is used.
See the competition pages for final rules, license, and scoring settings.
