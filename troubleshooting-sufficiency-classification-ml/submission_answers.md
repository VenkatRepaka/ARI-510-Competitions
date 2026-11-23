# Submission form answers (temporary draft)

Plain text, ready to paste into the form. Dev score = F1 of class 1 ("insufficient") over 5-fold × 5-repeat stratified CV on the 64 labelled questions. MPNet and the LLM checklist use a fixed cut-off of 0.5; the other runs and comparisons tune the cut-off inside each training part (see Dev set).

---

## Methodology

Approach: decide whether a tech-support forum question gives enough information to start troubleshooting (1 = insufficient). There are 64 labelled questions (22 insufficient) and 50 test questions. From the definition and the training examples, the label appears to depend on what is missing for the particular problem (a version number, the exact error text, a file the reader cannot see) rather than on the topic. Two lines of work were compared: conventional text models, and a local LLM whose Yes/No answers to plain questions become features for a small classifier.

Preprocessing: text kept as given ("Title: … Question: …"), no cleaning, since version numbers, quotes and punctuation are useful cues. Some models use the title and body separately. TF-IDF, scaling and PCA are fitted inside each training fold. Sentence encoders and LLMs are used without fine-tuning, so their outputs for all 114 questions are computed once and cached.

Dev set and evaluation: every model is scored with RepeatedStratifiedKFold (5 folds × 5 repeats, seed 42). Two cut-off procedures were used. MPNet and the LLM checklist were scored at a fixed cut-off of 0.5. For runs with threshold tuning (TF-IDF, the LLM wording average, the 91 non-LLM models and the per-prompt comparisons), the F1-maximising cut-off was selected on inner 4-fold cross-validation predictions within each training part and then applied to the held-out fold. The reference is predicting "insufficient" for every question (F1 0.512). A single model's fold-to-fold standard deviation was typically about 0.12 (0.04–0.23 across models); no paired test between models was run, so small differences are reported as observed. Out-of-fold AUC was also tracked. Test questions were never read or labelled.

Models tried:
- ALL-INSUFFICIENT: predict 1 for every question. Dev 0.512. Reference only.
- MPNET-LR: all-mpnet-base-v2 sentence embeddings + logistic regression C = 10, fixed cut-off 0.5. Dev 0.454. Submitted as run1.
- HAND-MADE: 15 regex cues (version, OS, device, error mentioned, quoted message, "tried", attachment or screenshot, how-to wording, steps) and 16 writing-style counts (length, digits, capitals, common words, vocabulary variety, technical tokens, links, paths, code), with logistic regression, naive Bayes, kNN, LDA, SVM, random forest, extra trees and gradient boosting. Dev 0.32–0.51; vocabulary variety alone 0.530. Not submitted.
- WORDS: TF-IDF of words or character n-grams, word/character counts with naive Bayes. Best: TF-IDF words 1–2 + logistic regression C = 10, dev 0.536 (AUC 0.60), cut-off 0.26 chosen on out-of-fold training predictions; predicts 29 of 50 test questions insufficient. Submitted as run4, the final choice.
- EMBEDDINGS: mpnet and BERT vectors of the full text, title, body, or title + body, with logistic regression, SVM, kNN, LDA and PCA + LR. Dev 0.22–0.52. Not submitted.
- NON-LLM VOTES: soft votes of hand-made, BERT and mpnet-body models (members fixed in advance). Dev 0.44–0.46. Not submitted.
In total 91 non-LLM models were compared (nonllm.ipynb); 5 scored above the reference, all by 0.03 or less.
- LLM-CHECKLIST: local Qwen2.5-VL-7B-Instruct (4-bit, MLX) answers 8 Yes/No questions (below). No text is generated: P(Yes) / (P(Yes) + P(No)) of the first answer token is the score, which is deterministic; for each checklist prompt, the average combined probability of Yes and No was at least 99.9 % (lowest single value 99.86 %). A logistic regression (standardised, C = 0.03, balanced) weighs the 8 scores, fixed cut-off 0.5. Dev 0.611 (0.589 with threshold tuning). Prepared (submission_llm_checklist.csv) but not submitted.
- LLM-WORDING-MEAN: the same model asked the main question in 8 wordings; the average goes into a 1-feature logistic regression. Dev 0.602. Prepared (submission_llm_wording_mean.csv) but not submitted.
- LLM + HAND-MADE / MLP: checklist scores plus the regex cues in a logistic regression (0.47), or in a small MLP (0.43). Not submitted.
- SECOND LLM: the same prompts on Qwen3-8B. Best AUC 0.74, but no single prompt beat Qwen2.5's best dev F1. Not submitted.

Hyperparameter selection: checklist LR C = 0.03 vs 0.3 (0.611 vs 0.575, both at cut-off 0.5), the stronger regularisation kept. TF-IDF C = 1 vs 10 (0.533 vs 0.536). Other non-LLM models use standard settings fixed in advance; the comparison was between inputs and model types. MPNet and the LLM checklist used a fixed cutoff of 0.5. For runs with threshold tuning, the cutoff was selected on inner cross-validation predictions.

LLM prompts: every prompt uses the same read-out and is scored so that higher = more likely insufficient, then compared on the 64 labelled questions (AUC and dev F1 of a 1-feature LR). 38 prompts in three sets, each on Qwen2.5 and Qwen3 (prompts.ipynb, logs/prompt_results.csv).

Checklist (system: "You are an experienced IT support engineer reviewing questions posted on a tech-support forum. A question is SUFFICIENT if it provides enough relevant details to understand the issue and begin troubleshooting; it is INSUFFICIENT if it lacks important details needed for effective troubleshooting. Answer only with Yes or No."):
1. "Does this question provide enough information to understand the issue and begin troubleshooting?"
2. "Would you have to ask the poster for more details before you could start helping?"
3. "Does the question name the specific product AND its version, device model or operating system?"
4. "Does the question clearly describe what goes wrong (or what the user wants to achieve) in concrete terms?"
5. "If an error or failure is mentioned, is the exact error message or behaviour quoted? (Answer Yes if no error is involved.)"
6. "Does the question say what the user has already tried or the steps that lead to the problem?"
7. "Does the question refer to a file, screenshot, data, formula or error that is NOT actually shown in the text?"
8. "Is the question vague, very short, or ambiguous about what exactly the problem is?"
On Qwen2.5, item 2 scored highest alone (dev 0.628, AUC 0.69); item 1 scored 0.464 and items 4–6 0.46–0.48.

Wordings of the main question: bare ("Is this question sufficient?"), with the competition's definition, reversed ("Does this question LACK important details …?"), support-engineer role, checklist-then-decide, "be strict", annotator framing ("label 1?"), a 1–5 scale, and few-shot with four labelled training examples (two per class) shown before the question; those four were excluded from that prompt's evaluation. On Qwen2.5 the reversed (AUC 0.72) and checklist-then-decide (0.71) wordings scored highest; bare (0.441), 1–5 scale (0.462) and few-shot (0.471) scored lowest. The average of eight wordings gave dev 0.602.

Plain-English set: 21 prompts with one template, 9 ways of asking the main question (e.g. "Is important information missing from this question?", "If you were answering this, would you need to ask the person for more details first?") and 12 single checklist items (product, symptom, error message, what was tried, file not included, vague, how-to, expected vs actual, when it started, scope, one problem, every time or sometimes). Highest: "would you need to ask … more details first?" on Qwen3 (AUC 0.74, dev 0.597) and "is important information missing?" on Qwen2.5 (AUC 0.72, dev 0.619). Several narrow items were near random (AUC 0.46–0.55). On average this set scored lower than the first attempt (mean AUC 0.57–0.58 vs 0.63–0.68).

On Qwen2.5, prompts asking what is missing scored higher than prompts asking whether the question is sufficient; on Qwen3 the pattern was mixed. The best single prompt differed between models (the follow-up question: 0.628 on Qwen2.5, 0.417 on Qwen3). Post length alone ranks the questions with AUC 0.62; the LLM "vague" score correlates with length (−0.64), "need more details" partly (−0.36), and "file not shown" and "product and version" hardly (+0.09, +0.17). These correlations describe the scores, not how the model arrived at them.

External data and models: no external data. Pretrained models, not fine-tuned: Qwen2.5-VL-7B-Instruct and Qwen3-8B (Apache-2.0, 4-bit MLX conversions), all-mpnet-base-v2 and bert-base-uncased (Apache-2.0). An AI coding assistant (Claude) helped write the code, draft and review the prompts, and run the experiments.

---

## Runs

run1 | MPNET-LR: mpnet sentence embeddings + logistic regression C=10 | 0.454 | 0.46153
run4 | WORDS: TF-IDF words 1-2 + logistic regression C=10 (final) | 0.536 | 0.52173

(Not submitted: LLM-CHECKLIST, dev 0.611 at cut-off 0.5 / 0.589 tuned, and LLM-WORDING-MEAN, dev 0.602 tuned.)

---

## Reflection

The final submission is TF-IDF + logistic regression (dev 0.536), the highest-scoring non-LLM model. Both LLM runs scored higher in local validation. The checklist had the highest reported local F1 at its fixed cutoff (0.611). With both LLM runs threshold-tuned, the wording average scored highest (0.602; checklist 0.589). Choosing TF-IDF was my decision to keep the submitted model non-LLM. I am skeptical of relying on an LLM for this judgement, and I preferred a model that learns only from the labelled questions and whose word weights I can inspect. This was a judgement call rather than a result of the experiments.

The conventional models all scored close to the all-insufficient reference. Run1's public score (0.462) was close to its CV score (0.454), but that is one comparison on about 25 questions; it does not validate the CV in general or predict the TF-IDF score. The prompt experiments showed that the LLM scores were sensitive to wording and to the model; they do not show how the LLM reached its answers.

Difficulties: with 64 questions, scores varied considerably between folds, so small differences were hard to interpret, and where the threshold was tuned it had to be done with a nested CV. The mlx-vlm library kept position information between prompts and had to be reset before every call. Running 38 prompts × 114 questions × 2 models on a laptop took several hours.

What I learned: on this dataset the LLM features scored higher locally than anything trained only on the 64 questions, but local score was not the only thing I weighed in choosing what to submit. I found it useful to test prompts on labelled data rather than judge them by how they read, and using one CV setup, fixed seeds and a simple reference line made the results easier to compare.
