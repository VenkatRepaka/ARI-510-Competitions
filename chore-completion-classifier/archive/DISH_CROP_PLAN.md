# Dish completion with local crop evidence

Test whether a small classifier that recognizes soiled dishes helps the current full-image RBF SVM catch dirty plates beside a clean sink. Start with the roughly 30 labeled dish photos already available for training.

## 1. Label the training crops

- Keep the competition label on each original photo. A sink clear of dirty dishes can be `Washing Dishes|Complete` with a clean plate in or next to it. A clean sink with visibly soiled plates beside it is `Washing Dishes|Incomplete`.
- Draw roughly one to three useful boxes per photo. For a clean sink next to two soiled plates, draw one box around the plates and one around the clean sink. Include enough margin to preserve food residue and the dish surface.
- Label crops `dirty_dishes_visible`, `no_dirty_dishes_visible`, or `unclear`. Exclude unclear crops from fitting. Include genuinely clean dishes as negatives, especially clean plates with patterns, shadows, or reflections. Keep examples of clean plates and dirty plates at similar scales.
- Draw boxes by hand for the first pass. If annotation becomes slow, use a frozen detector to propose plate, bowl, cup, sink, and rack boxes. Review its proposals and add any missed dishes manually.
- Keep a crop manifest with `parent_id`, `scene_group`, `x_min`, `y_min`, `x_max`, `y_max`, and `crop_label`. Store pixel coordinates in the EXIF-corrected original image and keep the original files intact.

## 2. Build the local classifier

1. Extract frozen DINOv2-L embeddings from the labeled crops and fit an L2-regularized logistic regression for visible dirty dishes. Start with L2-normalized embeddings and `C=1`.
2. At inference, score a fixed grid of overlapping tiles on each full photo. As a starting rule, use tiles about half the image width and height with a quarter-image stride in both directions. Apply the same rule to training, validation, and test photos when generating full-image features.
3. Start with the maximum tile score as the single added feature. Concatenate it with the existing full-image features and keep the existing SVM settings fixed for the first comparison. Save the highest-scoring tile for reviewing false positives and misses.

## 3. Validate by scene

- Freeze class-stratified, scene-grouped folds for the original photos. Keep a parent, its crops, and any alternate views of its scene in the same fold. Start with four folds if the groups allow class coverage.
- Within each outer fold, train the local classifier on labeled crops from outer-training scenes only. Generate tile scores for the outer-validation originals with that classifier. Generate tile scores for the outer-training originals with grouped inner cross-fitting, so the SVM trains on local scores from classifiers that have never seen those scenes. Fit any scaling using outer-training data.
- Compare the existing full-image SVM against the same SVM plus the tile score on identical folds. Evaluate one prediction per original photo. Report pooled out-of-fold four-class macro F1, the two dishes-class F1 scores, and the count of dish false positives and false negatives.
- Review paired errors by scene. Look for recovered incomplete scenes with soiled dishes beside a clean sink and new false alarms on clean plates. If the maximum score is repeatedly driven by a single misleading tile, test top-two averaging as a follow-up.

## 4. Decide and refit

- Keep the local feature if it improves predictions across multiple held-out scenes and the four-class comparison supports it. Retain the current SVM if the apparent improvement depends on one scene or causes too many clean-plate false alarms.
- If selected, generate grouped out-of-fold tile scores for all eligible training originals to fit the final SVM. Refit the local classifier on all eligible labeled crops for inference. Generate the submission with automatic tiles and save the model settings, crop manifest, split assignments, and out-of-fold predictions.
- If a separate dev set is already reserved, evaluate it after selecting the configuration. Keep competition test photos for automatic inference.
