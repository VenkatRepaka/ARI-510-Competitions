# Troubleshooting Sufficiency Classification

Code and results for the Kaggle class competition
[troubleshooting-sufficiency-classification-ml](https://www.kaggle.com/competitions/troubleshooting-sufficiency-classification-ml).

**Task.** Decide whether a tech-support forum question gives enough information to start troubleshooting.
Label `sufficiency`: 0 = sufficient, 1 = insufficient.
**Metric.** F1 of class 1 ("insufficient"), higher is better (the public scores fit binary F1 of class 1).

**Final submission.** TF-IDF of words and word pairs + logistic regression (`submission_tfidf.csv`):
dev F1 0.536, public 0.52173. This is a non-LLM model, chosen on purpose although the LLM models scored
higher in local validation (see section 7).

---

## 1. Data

| File | Rows | Columns |
|---|---|---|
| `data/train.csv` | 64 questions (22 insufficient, 42 sufficient) | `id, text, sufficiency` |
| `data/test.csv` | 50 questions | `id, text` |
| `data/sample_submission.csv` | 50 | `id, sufficiency` |

`text` has the form `Title: … Question: …`. Posts are 13–204 words long (median 73).
The data was downloaded from the competition's Data tab ("Download all files" → "Download as zip").

**Preprocessing** is minimal: the text is not cleaned, because version numbers, quotes, capitals and
punctuation are useful cues. Some models split the title from the body (`feats.split_title`).

---

## 2. How models are compared

All notebooks use the same evaluation (`feats.cv_eval`), so their numbers can be compared.

- **Repeated stratified cross-validation:** 5 folds × 5 repeats (`RepeatedStratifiedKFold`, seed 42),
  so every model is fitted 25 times. A single split of 64 questions would be far too noisy.
- **Cut-off.** Only a third of the questions are class 1, so the default cut-off of 0.5 is often not the
  best one for F1. Two procedures were used:
  - *fixed 0.5:* MPNet (run1) and the LLM checklist;
  - *tuned:* inside each training part an inner 4-fold CV gives out-of-fold probabilities, the cut-off with
    the highest F1 is chosen there, and it is applied to the held-out fold. The cut-off is never chosen on
    the questions it is scored on. Used for TF-IDF, the LLM wording average, all 91 non-LLM models and the
    per-prompt comparisons.
- **Reference line:** predicting "insufficient" for every question gives F1 **0.512**.
- **AUC** of the out-of-fold probabilities is also reported: it shows how well a score ranks the questions,
  whatever the cut-off (0.5 = random order).
- **Noise:** a single model's fold-to-fold standard deviation is typically about 0.12 (0.04–0.23). No
  paired test between models was run, so small differences are reported as observed.
- Learned steps (TF-IDF vocabulary, scaling, PCA) are fitted inside each training fold. Sentence encoders
  and LLMs are used without fine-tuning, so their outputs are computed once for all 114 questions and cached.
- The public board has only about 25 questions, so models were compared on dev F1, not on the public score.
  Test questions were never read or labelled.

---

## 3. Files, in the order they were built

### 3.1 `baseline.ipynb` — first baselines

Exploration of the data, then simple models (5×5 CV, cut-off 0.5):

| Model | Dev F1 |
|---|---|
| All insufficient (all 1) | 0.510 |
| All sufficient (all 0) | 0.000 |
| TF-IDF words + characters + logistic regression (C = 0.3–10) | 0.16–0.18 |
| all-mpnet-base-v2 embeddings + logistic regression C = 10 | **0.454** |

**Output:** `submission_baseline.csv` = **run1** (public 0.46153).
At a fixed 0.5 cut-off TF-IDF predicted almost nothing as insufficient; with a tuned cut-off (section 3.6)
the same kind of model scored 0.536.

### 3.2 `feats.py` — shared features and evaluation

- `handcrafted()` — 15 cues written after reading the training questions: length, version number,
  operating system, device model, error mentioned, quoted message, "tried / restarted", reference to an
  attachment or screenshot, how-to wording, numbered steps, first-person detail.
- `embed()` — mean-pooled, length-normalised sentence embeddings (mpnet, BERT), cached in `features/`.
- `best_threshold()`, `oof_proba()`, `cv_eval()` — the evaluation in section 2.

### 3.3 LLM scoring scripts

All three run a local LLM with MLX on the Mac (Qwen2.5-VL-7B-Instruct 4-bit, tag `q25`, and Qwen3-8B 4-bit,
tag `q3`, with thinking turned off). For every question and prompt the model runs **once**, without
generating text, and the probabilities of the first answer token give the score
`P(Yes) / (P(Yes) + P(No))`. This is deterministic. The share of probability on Yes + No is saved as
`<prompt>_mass` to check that the model followed the answer format (≥ 99.9 % on average per checklist prompt).

| Script | Prompts | Output |
|---|---|---|
| `llm_judge.py` | 8 checklist questions (enough information? need more details? product and version named? clear symptom? exact error quoted? steps tried? file or screenshot not shown? vague?) | `features/llm_<tag>.csv`, `features/llm_prompts_<tag>.json` |
| `prompt_variants.py` | 9 wordings of the main question: bare, with definition, reversed ("does it LACK…?"), engineer role, checklist-then-decide, strict, annotator, 1–5 scale, few-shot (4 labelled training examples, excluded from that prompt's evaluation) | `features/prompt_variants_<tag>.csv/.json` |
| `prompts_v2.py` | 21 plain-English prompts with one template: W1–W9 ways of asking the main question, C1–C12 single checklist items | `features/prompts_v2_<tag>.csv/.json` |

`run_v2_queue.sh` ran `prompts_v2.py` on both models one after the other (under `caffeinate`).
The exact prompt texts are in the `.json` files and in `submission_answers.md`.

**mlx-vlm fix:** the library kept position information from the previous prompt, which gave wrong
probabilities; `lm._rope_deltas` and `lm._position_ids` are reset before every call.

### 3.4 `prompts.ipynb` (built by `make_prompts_nb.py`) — which prompts worked

Each of the 38 prompts × 2 models is scored on the 64 labelled questions: AUC, F1 at 0.5, share predicted
insufficient, and dev F1 of a 1-feature logistic regression (tuned cut-off). Verdict: *worked* = AUC ≥ 0.6
and dev F1 > 0.51; *weak* = AUC ≥ 0.55; otherwise *did not work*. Results: `logs/prompt_results.csv`.

| Prompt | Qwen2.5 AUC / F1 | Qwen3 AUC / F1 |
|---|---|---|
| "Would you have to ask the poster for more details…?" (checklist) | 0.69 / **0.628** | 0.69 / 0.417 |
| "Is important information missing from this question?" (plain set) | **0.72** / 0.619 | 0.56 / 0.450 |
| "…would you need to ask the person for more details first?" (plain set) | 0.57 / 0.504 | **0.74** / 0.597 |
| "Does this question LACK important details…?" (reversed wording) | 0.72 / 0.540 | 0.69 / 0.543 |
| "Is this question sufficient?" (bare) | 0.58 / 0.441 | 0.53 / 0.510 |
| 1–5 completeness scale | 0.65 / 0.462 | 0.60 / 0.484 |
| Few-shot (4 examples) | 0.63 / 0.471 | 0.51 / 0.480 |
| Narrow items: error message, what was tried, one problem, every time or sometimes | 0.46–0.55 AUC | 0.46–0.55 AUC |
| Average of wordings v1–v8 | 0.70 / 0.609 | 0.71 / 0.568 |

Observations: on Qwen2.5, prompts asking what is *missing* scored higher than prompts asking whether the
question is *sufficient*; on Qwen3 the pattern was mixed. The best single prompt differed between the two
models. The plain-English rewrites did not score higher on average (mean AUC 0.57–0.58 vs 0.63–0.68 for the
first attempt). Post length alone ranks the questions with AUC 0.62, and the "vague" score correlates
with length (−0.64).

### 3.5 `models.ipynb` (built by `make_models_nb.py`) — first model comparison

| Model | Dev F1 tuned | Dev F1 @0.5 |
|---|---|---|
| Hand-made cues + logistic regression C = 0.03 | 0.487 | 0.390 |
| mpnet embeddings + logistic regression | 0.449 | 0.426 |
| BERT embeddings + logistic regression | 0.496 | 0.501 |
| BERT embeddings + SVM | 0.510 | 0.048 |
| LLM: "enough information?" only | 0.458 | 0.441 |
| **LLM checklist (8 scores) + logistic regression C = 0.03** | 0.589 | **0.611** |
| LLM checklist + logistic regression C = 0.3 | 0.536 | 0.575 |
| LLM checklist + hand-made cues | 0.473 | 0.496 |
| MLP (16 units) on LLM checklist + hand-made cues | 0.429 | 0.350 |

### 3.6 `nonllm.ipynb` (built by `make_nonllm_nb.py`, extra features in `nonllm_feats.py`) — 91 non-LLM models

`nonllm_feats.py` adds 16 writing-style counts (length, share of digits / capitals / common words,
vocabulary variety, technical tokens such as `0x80070005` or `report.xlsx`, links, paths, code, greeting…)
and title-only / body-only embeddings. All with tuned cut-off. Results: `logs/nonllm_results.csv`.

| Group | What was tried | Dev F1 range | Best |
|---|---|---|---|
| A. One number | single feature + learned cut-off | 0.41–0.53 | vocabulary variety 0.530 (AUC 0.65) |
| B. Hand-made | 15 cues / 16 counts / both × 9 classifiers (LR L2 and L1, naive Bayes, kNN, LDA, SVM, random forest, extra trees, gradient boosting) | 0.32–0.51 | text counts + SVM 0.514 |
| C. Words | TF-IDF words or characters, counts + naive Bayes, function words only | 0.28–0.54 | **TF-IDF words 1–2 + LR C = 10: 0.536** |
| D. Embeddings | mpnet / BERT × full, title, body, title + body × 6 classifiers | 0.22–0.52 | mpnet body + PCA 10 + LR 0.521 |
| E. Votes | soft votes of models from different groups (members fixed in advance) | 0.44–0.46 | 0.459 |

Only 5 of the 91 scored above the all-insufficient reference, each by 0.024 or less.

### 3.7 `make_submission.py` — submission files

Rebuilds every LLM and TF-IDF submission from the cached features (fixed seed; rerunning gives identical
files).

| File | Model | Test predicted insufficient |
|---|---|---|
| `submission_llm_checklist.csv` | LLM checklist + LR C = 0.03, cut-off 0.5 | 19 / 50 |
| `submission_llm_wording_mean.csv` | mean of wordings v1–v8 + 1-feature LR, cut-off 0.51 | 23 / 50 |
| `submission_tfidf.csv` | TF-IDF words 1–2 + LR C = 10, cut-off 0.26 | 29 / 50 |

(`submission_baseline.csv` comes from `baseline.ipynb`.)

---

## 4. Runs

| Run | File | Model | Dev F1 | Public |
|---|---|---|---|---|
| run1 | `submission_baseline.csv` | mpnet embeddings + LR C = 10, cut-off 0.5 | 0.454 | 0.46153 |
| run4 | `submission_tfidf.csv` | TF-IDF words 1–2 + LR C = 10, tuned cut-off (**final**) | 0.536 | 0.52173 |
| — | `submission_llm_checklist.csv` | LLM checklist (not submitted) | 0.611 @0.5 / 0.589 tuned | — |
| — | `submission_llm_wording_mean.csv` | LLM wording average (not submitted) | 0.602 tuned | — |

---

## 5. How to reproduce

From this folder, with the shared environment `../.venv` (Python 3.13; scikit-learn, pandas, torch,
transformers, mlx, mlx-lm, mlx-vlm, jupyter). LLM steps need an Apple-silicon Mac and the cached models.

```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace baseline.ipynb
```

```bash
../.venv/bin/python llm_judge.py && ../.venv/bin/python llm_judge.py mlx-community/Qwen3-8B-4bit q3
```

```bash
../.venv/bin/python prompt_variants.py && ../.venv/bin/python prompt_variants.py mlx-community/Qwen3-8B-4bit q3
```

```bash
../.venv/bin/python prompts_v2.py && ../.venv/bin/python prompts_v2.py mlx-community/Qwen3-8B-4bit q3
```

```bash
for n in prompts models nonllm; do ../.venv/bin/python make_${n}_nb.py && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace ${n}.ipynb; done
```

```bash
../.venv/bin/python make_submission.py
```

The LLM steps took several hours on a laptop (about 11–22 s per question with all prompts); their outputs
are already cached in `features/`, so the notebooks and `make_submission.py` run in minutes.

---

## 6. Models and data used

No external data. Pretrained models, all used without fine-tuning: Qwen2.5-VL-7B-Instruct and Qwen3-8B
(Apache-2.0, 4-bit MLX conversions from mlx-community), all-mpnet-base-v2 and bert-base-uncased
(Apache-2.0). An AI coding assistant (Claude) helped write the code, draft and review the prompts, and run
the experiments.

---

## 7. Final choice

TF-IDF + logistic regression (dev 0.536, public 0.522) is the final submission, the highest-scoring of the
non-LLM models. Both LLM models scored higher in local validation: the checklist had the highest reported
local F1 at its fixed cut-off (0.611); with both LLM runs threshold-tuned, the wording average scored
highest (0.602; checklist 0.589). Choosing TF-IDF was a decision to keep the submitted model non-LLM, out of
skepticism about relying on an LLM for this judgement and a preference for a model that learns only from the
labelled questions and whose word weights can be inspected. It is a judgement call, not a result of the
experiments. The full write-up for the submission form is in `submission_answers.md`.
