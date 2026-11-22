# What Makes Title Interesting — Phase 2b write-up

Competition: https://www.kaggle.com/competitions/what-makes-title-intresting

Metric: Mean Absolute Error (lower is better). Target: perceived interest of a content title, 1–6.

---

## 2. Approaches

### 2.1 Understanding the data (and why it shaped everything)

The training set has 120 titles (30 each from YouTube, Reddit, podcasts and news articles) with columns
`title_id, title, domain, interest`. The test set has 30 titles (7–8 per domain). There are no extra columns
(the annotation traits mentioned in the overview — curiosity gap, emotional language, exaggeration, direct
address, missing details — are not provided).

The most important detail is in the overview: **each training title was rated by one annotator, but each
test title's label is the mean of six annotators.** Training labels are therefore integers spread over the
whole scale (30 ones, 20 twos, 15 threes, 25 fours, 19 fives, 11 sixes; mean 3.13, sd 1.68), while test labels
are averages that sit much closer to the middle. Two consequences:

1. My dev MAE (against single-rater labels) is much higher than the leaderboard MAE (against 6-rater means)
   for the same model — e.g. the constant prediction scores 1.495 on dev but 0.848 on the public board. Dev
   scores are only useful for *comparing* models, not as an estimate of the leaderboard score.
2. Single ratings are very noisy. Using the constant's public score to calibrate a simple noise model, I
   estimated that only about 28% of the variance in a single rating is "real" title signal (true-score sd
   ≈ 0.9, single-rater noise sd ≈ 1.4). A model trained only on these 120 labels has very little signal to
   learn from — which turned out to be the central problem of this task.

**Preprocessing.** Minimal: no cleaning of the title text (capitalisation, emoji, hashtags and punctuation are
themselves cues). The domain was one-hot encoded and used in every model. For text encoders the input was
`"<domain>: <title>"`. For LLM prompts the domain was turned into a phrase ("YouTube video", "Reddit post",
"podcast episode", "news article").

### 2.2 Dev-set evaluation

With only 120 rows, a single train/dev split is too noisy, so every model was evaluated with the **same
5-fold cross-validation repeated 10 times** (`RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)`,
50 fits per model), reporting mean MAE ± sd over folds. All fitting (scalers, ridge, SVR, MLPs, fine-tuning)
happens inside each fold. Fold-to-fold sd is about ±0.18, so differences under ~0.02 are noise; I also
report the out-of-fold correlation with the labels, which is less noisy than MAE. Predictions are clipped
to [1, 6]. All seeds are fixed (42) so every run is reproducible.

Hyperparameters (ridge alpha, SVR C, kNN k, MLP epochs/weight decay, shrinkage factor) were chosen from small
grids on this CV. I also tested "shrinking" predictions toward the mean (`mean + s·(pred − mean)`), since the
test labels are compressed. On CV the best s was close to 1 and the gain was noise-level, so after run2 I
fixed s = 1.

### 2.3 Baselines (`baseline.ipynb`)

| Model (5×10 CV) | Dev MAE |
|---|---|
| Constant (train mean 3.13) | 1.495 |
| Per-domain mean | 1.478 |
| Hand-crafted features (length, caps, ?, !, digits, "you", …) + ridge | 1.494 |
| all-mpnet-base-v2 embeddings + ridge | 1.480 |
| mpnet embeddings + RBF SVR (C = 0.3) | 1.466 (1.463 with shrink s = 0.4) |
| mpnet embeddings + kNN (k = 40, cosine) | 1.469 |
| mpnet embeddings + MLP (PyTorch, 64 hidden) | 1.72–1.79 (overfits) |

The embedding SVR became **run2**. Everything trained only on the labels was barely better than the mean.

### 2.4 External data: clickbait transfer (`clickbait.ipynb`, never submitted)

Idea: the overview's traits (curiosity gap, exaggeration, direct address) are what clickbait detectors
measure, and a large dataset rated by several annotators could teach those cues. I used the **Webis
Clickbait Corpus 2017** (Zenodo, CC BY 4.0): 40,856 news tweets, each rated by 5 crowd workers on a 0–1
clickbait scale. Only the text/label files were downloaded (`fetch_webis.py`).

- Analysis: a single Webis rater agrees with the item mean much less than the 5-rater mean does
  (ICC 0.38 vs 0.75), the same noise problem as our task. Clickbait cues were "you", "this/these",
  "why/how", questions and number-first headlines.
- Trained on the official train split and tested on the official test split: TF-IDF ridge MSE 0.042 /
  Pearson 0.68; mpnet embeddings + ridge MSE 0.036 / Pearson 0.73; **mpnet + MLP (neural net)** MSE 0.036 /
  Pearson 0.72 — on par with the best 2017 Clickbait Challenge systems.
- Transfer: the frozen clickbait scorers were applied to our titles and used as features. Clickbait score
  was essentially uncorrelated with interest within each domain (Spearman −0.26 to +0.14), and the best
  CV was 1.468–1.473 (no gain). **Clickbait ≠ interesting**, so this was dropped. (Training on external data
  is also a grey zone under the competition rule "use train.csv for training".)

### 2.5 Non-LLM models (`nonllm.ipynb`, plus pure scikit-learn checks)

To be sure I was not overcomplicating things, I tried a broad set of models with no generative LLM:

| Model (+ domain) | Dev MAE |
|---|---|
| 24 hand-crafted features × domain interactions, ridge (α = 300) | **1.449** |
| Hand-crafted features, SVR | 1.451 |
| Character TF-IDF (2–5-grams), ridge | 1.457 |
| Word TF-IDF, ridge | 1.469 |
| Random forest / gradient boosting / kNN on hand-crafted features | 1.48–1.51 |
| bert-base-uncased mean-pooled + SVR (C = 1) | 1.433 |
| CLIP ViT-H-14 text encoder + kNN | 1.441 |
| mpnet (ridge / SVR / kNN) | 1.47–1.48 |
| **Fine-tuned mpnet** (all weights + regression head, 6 epochs, lr 2e-5) | 1.500 |
| Ensemble: best model of each family | 1.439 |

The hand-crafted features showed that cues mean different things per domain: longer, more descriptive
podcast titles were rated more interesting (Spearman 0.46), Reddit questions / how-why titles less
interesting (≈ −0.4), and hashtags, emoji and brackets lowered interest on YouTube. Fine-tuning a full
transformer on ~96 noisy titles per fold simply memorised noise. The best non-LLM CV (1.433) was picked
from 27 models, so it is optimistic.

### 2.6 Local LLM zero-shot (`llm_rate.py`, `llm.ipynb`)

Since the test label is an "average reader" judgement, I tried asking an LLM to act as an average reader.
Model: **Qwen2.5-VL-7B-Instruct (4-bit, Apache-2.0)**, run locally with mlx-vlm on my laptop. The LLM never
sees any labels, so its scores can be computed once for all 150 titles and used inside CV without leakage.

**Readout.** Instead of sampling an answer, I read the model's next-token probabilities for the digits 1–6
and took the expected value. This gives a smooth score (e.g. 3.74) instead of an integer, and is
deterministic.

**Prompt selection.** The main prompt mirrors the task statement:

> How interesting would a typical reader find this {kind} title?
> Scale: 1 = not interesting at all, 2 = slightly, 3 = somewhat, 4 = fairly, 5 = very, 6 = extremely interesting.
> Title: {title}
> Answer with a single digit from 1 to 6.

with the system prompt "You are an average adult internet user judging content titles." Other prompts I
tried (all saved in `features/*prompts*.json`):

- a **click-likelihood** prompt ("how likely are you to click … 1–6");
- five **yes/no trait prompts** matching the annotation guidelines (curiosity gap, emotional language,
  exaggeration, direct address, missing details) → P(Yes);
- **six reader personas** (student, parent, retiree, professional, casual browser, skeptical reader) with a
  first-person version of the rating prompt, averaged to simulate a panel of 6;
- **pairwise comparisons** (section 2.8).

**Calibration model.** The raw LLM score is on the wrong scale (raw MAE 1.506), so I fit a small ridge
regression (α = 1) on [LLM score + domain one-hots] — five learned numbers.

| LLM features (+ domain, ridge) | Dev MAE |
|---|---|
| **Interest rating only** | **1.415** (run3) |
| + click rating | 1.428 |
| + 5 trait scores | 1.443–1.454 |
| + full 1–6 probability vectors | 1.449+ |
| 6-persona average | 1.446 (personas correlate 0.83 with each other) |
| Raw LLM score, no fit | 1.506 |
| **MLP (16 hidden, dropout, 5 seeds) on LLM ratings + traits + domain** | 1.463 (neural net) |
| mpnet embeddings + LLM score, SVR | 1.457 |

The single "typical reader" prompt was best; extra prompts mostly added noise. I also tried many ways of
fitting the LLM score — per-domain slopes (1.418–1.423), quadratic term (1.418), isotonic (1.428), quantile
regression (1.431), Bayesian ridge (1.442), Gaussian process (1.501), gradient boosting (1.406, within noise),
annotator-block effects inferred from `title_id` order (1.416), blending with the BERT model (1.411–1.414). None
reliably beat the plain 5-parameter ridge.

**A bug worth reporting.** In the first run, the probability mass on the digits averaged only 0.63. I traced
it to mlx-vlm caching the rotary-position ids from the first prompt and silently reusing them for later
text-only prompts. After resetting that state before every call, the mass was ≥ 0.999 for every title, and all
results above use the fixed scores.

### 2.7 A second local LLM (Qwen3-8B)

Qwen3-8B (4-bit, Apache-2.0, thinking disabled) with the same prompt ranked titles slightly better (Spearman
0.23 vs 0.19, and positive on news articles) but was over-confident (most mass on 4–5), so its ridge CV was
worse (1.439). Its ratings correlate only 0.66 with Qwen2.5's, so I averaged the two z-scored ratings:
dev MAE 1.414 with the best out-of-fold correlation so far (0.233). Not submitted.

### 2.8 Pairwise comparisons (`llm_pairwise.py`)

People (and LLMs) are often more consistent at "which of these two is more interesting?" than at absolute
scales. Each title was compared with ≥ 12 random others (1,126 pairs), and **every pair was asked in both
orders** to cancel position bias. The soft win probability came from P("A") vs P("B"); I turned the
comparisons into one score per title as a win rate and as a Bradley–Terry score (L2-regularised).

- Qwen2.5: consistent across orders (r = 0.86), mild first-position bias (45%). Win rate + domain:
  **dev MAE 1.393** (run8); Bradley–Terry 1.408.
- Qwen3-8B: inconsistent across orders (r = 0.63) and position-biased (57%) → 1.473; averaging it in hurt.

run8 had the best local dev score, but it scored **worse** on the public board (0.812 vs 0.724 for run3).
The dev gain (0.022) was within noise and slightly optimistic (best of two read-outs), and the public set is
only ~15 titles, so I treat pairwise as "not clearly better".

### 2.9 Frontier LLM zero-shot: Claude Opus 5.5 (`claude_rate.py`)

Because every step up in LLM quality had helped (embeddings → 7B LLM), I tried a much larger hosted model,
**Claude Opus 5.5**, through the official Anthropic SDK (code calls the API for each of the 150 titles; no
manual input). The Claude API does not return token probabilities, so to keep a smooth score I asked for a
**probability distribution** instead, enforced with a JSON schema (structured output):

> How interesting would a typical reader find this {kind} title? Scale: 1 = not interesting at all … 6 =
> extremely interesting. Title: {title}. Different readers disagree, so give the probability that a randomly
> chosen typical reader would pick each score (p1..p6, each between 0 and 1, summing to 1).

System prompt: "You are simulating a panel of average adult internet users judging how interesting content
titles are. Judge only from the title and its source type." Effort was set to "low". The probabilities were
normalised and converted to an expected rating, then calibrated with the **same 5-parameter ridge** as run3.
Total cost was about $0.46. Because the API has no random seed, the run is reproduced from the saved ratings
in `features/claude_scores.csv` (the ridge step is deterministic).

| Signal (+ domain, ridge α = 1) | Spearman with labels | Dev MAE |
|---|---|---|
| Qwen2.5-7B rating (run3) | 0.19 | 1.415 |
| **Claude Opus 5.5 rating (run9)** | **0.38** | **1.378** |
| Claude, no domain | — | 1.384 |
| Claude + Qwen2.5 averaged | 0.32 | 1.381 |
| Claude + Qwen2.5 + Qwen3 averaged | 0.31 | 1.384 |

Claude's rating correlated twice as strongly with the labels as either local model and was positive in every
domain, including news articles (article 0.25, podcast 0.46, Reddit 0.38, YouTube 0.31), where the local models
were at or below zero. Averaging the local models in did not help.

### 2.10 Neural networks trained (assignment requirement)

- PyTorch MLP on mpnet embeddings (baseline notebook): 1.72–1.79 (overfits).
- PyTorch MLP on Webis embeddings for clickbait regression: Pearson 0.72 on the Webis test set.
- Fine-tuned mpnet (all weights): 1.500.
- PyTorch MLP on LLM features (16 hidden, dropout 0.2, weight decay, 5-seed average): **1.463** — beats
  the constant and the embedding baseline (prepared as run4, `submission_llm_mlp.csv`).

---

## 3. Runs

Dev MAE = 5-fold × 10-repeat CV against single-annotator train labels (so it is much higher than the
leaderboard MAE, which uses 6-annotator means). Lower is better.

| Run | Approach | Dev MAE | Public MAE |
|---|---|---|---|
| run1 | Constant: train mean 3.133 | 1.495 | 0.84793 |
| run2 | all-mpnet-base-v2 embeddings + domain, RBF SVR C = 0.3, shrink s = 0.4 | 1.463 | 0.82826 |
| run3 | Zero-shot Qwen2.5-VL-7B "typical reader" rating (expected value of digit probabilities) + domain, ridge α = 1 | 1.415 | 0.72446 |
| run8 | Pairwise comparisons with Qwen2.5-VL-7B (≥ 12 opponents, both orders), win rate + domain, ridge α = 1 | 1.393 | 0.81173 |
| run9 | Zero-shot Claude Opus 5.5 rating (structured P(1..6) → expected value) + domain, ridge α = 1 | **1.378** | **0.62883** |

Host baselines (public): `benchmark_mean.csv` **0.84800**; `sample_submission.csv` (3.5 for every title)
**0.92133**. All of run2, run3, run8 and run9 beat both baselines; run1 ties the mean baseline (it is the same
idea).

Prepared but not submitted: run4 MLP on LLM features (dev 1.463), run5 bert-base + SVR (1.433), run6 average of
Qwen2.5 + Qwen3 ratings (1.414), run7 pure scikit-learn hand-crafted features × domain ridge (1.449).

Final selections: **run9** (best dev score) and **run3** (best local-only model).

---

## 4. What worked

**Best on dev: run9, the zero-shot Claude Opus 5.5 rating calibrated by a 5-parameter ridge (dev MAE 1.378),**
which was also best on the public board (0.629).

Why I think it worked:

1. **The labels are the bottleneck, not the model.** Each training title has one rater's opinion, and I
   estimated that most of a single rating is noise. Anything learned only from the 120 labels — TF-IDF,
   hand-crafted features, embeddings, fine-tuned transformers, tree models, MLPs — landed between 1.43 and
   1.50, barely better than predicting the mean (1.495). The gains had to come from outside the labels.
2. **An LLM supplies the "average reader" judgement the test set measures.** The test label is an average of
   six people. A zero-shot LLM was never trained on our noisy labels; it brings a prior about what people in
   general find interesting, which is exactly what a 6-rater mean measures. That is why the LLM runs improved
   the leaderboard far more than dev MAE suggested (dev is measured against single noisy ratings).
3. **Model quality mattered more than modelling tricks.** Scores improved with each step up in model quality:
   sentence embeddings (public 0.828) → 7B local LLM (0.724) → Claude Opus 5.5 (0.629). In contrast, extra
   prompts, personas, trait scores, fancier calibration models and ensembles changed dev MAE by less than ±0.01.
   Claude also fixed the local models' blind spot on news articles.
4. **Keep the learned part tiny.** With noisy labels, the best use of the 120 titles was to learn only the
   scale and a per-domain offset (one slope + four intercepts). More flexible models fitted noise.
5. **Read-outs that keep uncertainty helped.** Using the expected value of a probability distribution
   (digit probabilities for local models, a requested distribution for Claude) gave smoother, better-calibrated
   scores than a single sampled answer.

What did not work, and why: clickbait transfer (clickbait is a different construct from interest), fine-tuning
(memorises single-rater noise), personas (same model → highly correlated answers), Qwen3 pairwise (inconsistent
across orders), and pairwise in general on the public board (dev gain within noise). The public board has only
~15 titles (standard error ≈ ±0.12), so I chose final submissions by dev results, as the assignment asks.
