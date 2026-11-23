# Submission form answers (temporary draft)

Plain text, ready to paste into the form. Dev score = F1 (artifact present) on the host's validation split
(118 images), with the model fitted on train only, threshold 0.5. The CV F1 (5-fold × 3-repeat stratified CV on
train + validation) is given alongside, because it was used to choose between models.

---

## Methodology

Approach: the task is to decide whether a photo shows a camera artifact (obstruction, exposure fault, lens contamination, blur, or lens fog). The clean class contains hard negatives such as weather fog, in-focus raindrops on a window, shallow depth of field and dark night scenes, so low-level sharpness or brightness alone cannot separate the classes; the model needs to know what the scene shows. With only 941 labelled images, the plan was to use frozen pretrained vision models as feature extractors, add hand-crafted image-quality statistics for blur and exposure, and train small classifiers on top. A neural network (MLP) was trained on the same features, and an error review guided one more round of features.

Preprocessing: each image is loaded with PIL, rotated by its EXIF orientation and converted to RGB, as the host asks. Nothing else is cleaned. For the embeddings, each image is resized by the model's own transform and encoded twice (as is and mirrored), and the two embeddings are averaged and L2-normalised. For the quality statistics, the image is downscaled so its long side is 1024 px. image_id and filenames are never used as features. All features are standardised inside each fold.

Dev set and evaluation: the host's split is used as intended: fit on train (823), score on validation (118) = dev F1 at threshold 0.5. With 118 images one image moves F1 by about 0.01, so every model is also scored with 5-fold stratified CV repeated 3 times (seed 0) on train + validation (941 images); CV F1 is the mean over the 3 repeats, and its spread (about ±0.003–0.007) was used as the noise level. Model choices were made on CV F1, with dev F1 as a second check, never on the public score. The best threshold on the out-of-fold probabilities was also checked; it was 0.41–0.48 for the good models, close to 0.5, so all submissions use 0.5. All seeds are fixed and every script gives identical outputs when rerun. Test images were only used for the final prediction; they were never viewed or labelled.

Models tried:

- ALL-ONES: predict "artifact" for every image. Dev 0.703. Reference only, not submitted.
- QUALITY-STATS: 74 hand-crafted statistics computed on the whole image and on a 3×3 grid (min / max / spread across cells, border minus centre): sharpness (Laplacian variance, gradients, Tenengrad, edge density, gradient anisotropy for motion blur), frequency content, exposure (brightness percentiles, clipped highlights and shadows, dynamic range, entropy), colour and haze (saturation, dark channel, RMS and local contrast). RBF SVM: dev 0.776, CV 0.780. Not submitted.
- SINGLE-BACKBONE: frozen embeddings from one pretrained model, RBF SVM. DINOv2-L (448 px, CLS + mean patch token): dev 0.790, CV 0.853. SigLIP SO400M: dev 0.898, CV 0.882. EVA02-L CLIP: dev 0.873, CV 0.880. The image–text (CLIP-style) models beat DINOv2, probably because they encode concepts such as "foggy", "blurry" or "seen through a window". Not submitted.
- BACKBONE+QUALITY: one backbone plus the 74 statistics, RBF SVM. Best: EVA02 + quality, dev 0.889, CV 0.886; SigLIP + quality, dev 0.915, CV 0.880. Not submitted.
- ALL3-SVM: DINOv2-L + SigLIP + EVA02-L embeddings + 74 quality statistics, standardised, RBF SVM (C = 3). Dev 0.906, CV 0.886 ± 0.003. Logistic regression (0.848 / 0.867) and gradient boosting (0.816 / 0.866) on the same features were worse. Submitted as run 1.
- MLP: PyTorch network on the frozen features: standardise → input dropout → Linear(D, 256) → GELU → dropout 0.5 → Linear(256, 1). Five networks with different random initialisations are trained together as one batched module and their probabilities averaged. AdamW, cosine learning-rate schedule, batch 64, a fixed 60 epochs (no early stopping, so no held-out data is used in training). Three settings (default; 64 hidden units; stronger regularisation with input dropout 0.5 and weight decay 0.1) on three feature sets. Best: EVA02 + quality with strong regularisation, dev 0.873, CV 0.880. Averaging MLP and SVM probabilities gave 0.883–0.893 vs 0.890 for the SVM alone (within noise). Submitted as run 2.
- REGION-FOG-FEATURES (after the error review): EVA02 embeddings of a 3×3 grid of overlapping tiles, summarised as 19 tile-similarity scalars or as aggregate tile embeddings, plus quality statistics on the top / middle / bottom thirds of the image. Idea: lens fog veils the whole frame evenly, while weather fog fades with distance, so the bottom of the frame stays clear. Alone these reach CV 0.70–0.74, but added to the best models they gave 0.884–0.885 (no change) or 0.867–0.875 (worse). Not submitted.

Hyperparameter selection: the classical models were not tuned; they use standard settings fixed before any results were seen (SVM C = 3 with default gamma; logistic regression C = 0.1; gradient boosting 300 iterations at learning rate 0.05), and the comparison was between feature sets and model types rather than settings. For the MLP, three settings were compared on the same CV: 256 vs 64 hidden units, and input dropout 0.2 / weight decay 0.01 vs input dropout 0.5 / weight decay 0.1. The strongly regularised 256-unit network was best on EVA02 + quality; the other settings were within noise. Epochs were fixed at 60 in advance. Threshold kept at 0.5 (see Dev set).

Error review: run1's out-of-fold predictions on all 941 images give 61 missed artifacts and 50 false alarms. Contact sheets of the most confident errors showed two main confusions: fog/mist landscapes (labelled artifact when the fog seems to be on the lens, clean when it is in the scene, and the two often look the same) and raindrops on windows. A near-duplicate search (DINOv2 cosine > 0.95) found 14 pairs, three of them identical images with opposite labels (a rainy window, a misty forest, a foggy field; one copy is in validation). No model can get both copies right, so these set a small ceiling on F1. The labels were left unchanged, because relabelling by eye would add my own judgement to the data. Two training images are near-copies of test images; nothing was done about them.

External data and models: no external data. Pretrained models with frozen weights: DINOv2 ViT-L/14 (Apache-2.0), SigLIP SO400M (Apache-2.0), EVA02-L CLIP (MIT). Competition data: MIT. An AI coding assistant (Claude) helped write the code and run the experiments.

---

## Runs

run 1 | ALL3-SVM: DINOv2-L + SigLIP + EVA02-L embeddings + quality stats, RBF SVM | 0.906 | 0.88059  
run 2 | MLP: EVA02-L embeddings + quality stats, PyTorch MLP ensemble | 0.873 | 0.88188

(CV F1: run1 0.886, run2 0.880.)

---

## Reflection

ALL3-SVM had the best dev F1 (0.906) and tied for the best CV F1 (0.886), so it is the final choice. The main lesson was that the choice of pretrained model mattered far more than the classifier on top: models trained on images paired with text (SigLIP, EVA02) beat a stronger self-supervised model (DINOv2) by about 0.03 CV, because telling lens fog from weather fog, or drops on the lens from rain on a window, is a question of what the scene shows rather than how sharp it is.

The neural network did not beat the SVM. With about 750 training images per fold, a regularised MLP ensemble reached CV 0.880 against 0.886, and averaging the two did not help. I kept the MLP as run2 because it is the best neural model, not because I expected it to win.

The error review changed how I read the scores. Most remaining errors are fog and raindrop scenes that are labelled both ways, and three identical images carry opposite labels, one of them in validation. The region-based fog features designed from this review carried some signal on their own but added nothing to the full-image embeddings, so I stopped adding features: the remaining errors look like label ambiguity more than missing information.

Difficulties: the first MLP trained its five networks one after another and needed about a minute per fit, which made the 15-fold CV too slow; training all five as one batched module cut this to 14 seconds with the same results. Dev F1 on 118 images is noisy (one image ≈ 0.01), so repeated CV on all 941 labelled images was needed to tell models apart.

What I learned about doing this task well:

Look at the data before modelling. Viewing sample images of each class showed that the clean class contains hard negatives (weather fog, in-focus raindrops on a window, shallow depth of field, dark night scenes). That changed the task from "measure image quality" to "decide whether the camera caused the problem", and it is why models that understand the scene did best.

Image-quality measurement was new to me. To understand the hand-crafted features I had to learn how classic image processing describes a photo: the variance of the Laplacian and Sobel gradients (Tenengrad) as sharpness measures, Canny edge density, the share of high-frequency energy in the Fourier spectrum as a blur cue, clipped highlights and shadows for exposure faults, and the dark channel, saturation and local contrast as haze cues. Computing them on a 3x3 grid as well as the whole image catches problems that cover only part of the frame, such as a finger at the edge. These features alone reached CV F1 0.78, and they added a little on top of the embeddings, but they cannot tell lens fog from weather fog. I also learned small practical details that matter, such as honouring the EXIF orientation tag (158 of 1,177 images were stored rotated, mostly sideways) and that image sizes are written width x height while numpy arrays are rows x columns.

With under 1,000 labelled images, frozen pretrained models plus a simple classifier are a strong approach; training a large network from scratch was not needed.

Ensembles in an MLP were new to me. Training several copies of the same network from different random starting weights and averaging their probabilities gives more stable predictions than one network. I also learned that the copies can be stored as stacked weights and trained together with one batched matrix multiplication, which gives the same results as training them one by one but was about 4x faster. The result above also showed that a neural network is not automatically the best choice on a small dataset.

Evaluate carefully. One validation image changes F1 by about 0.01, so I used 5-fold cross-validation repeated 3 times on all labelled images, chose models on that, fixed every random seed, and checked that each submission file is reproduced exactly when the code is rerun.

Review the mistakes, not only the score. The error review explained where the score stops, showed that the errors were not simply the blurriest or darkest images, and told me when to stop adding features.

Keep experiments organised. Every approach was a set of saved feature files plus one classifier, run by the same script and recorded in the same results tables, which made it easy to compare approaches and to rerun any of them.
