# What Makes a Title Interesting

Code and results for the Kaggle class competition
[what-makes-title-intresting](https://www.kaggle.com/competitions/what-makes-title-intresting).

**Task.** Predict how interesting a content title is, on a scale from 1 to 6, from the title text and its
source (YouTube, Reddit, podcast or news article).
**Metric.** Mean Absolute Error (MAE), lower is better.

**Best result.** Zero-shot rating by Claude Opus 5.5, mapped onto the label scale by a small ridge regression:
public MAE **0.629** (`submission_claude.csv`).

---

## 1. Data

| File | Rows | Columns |
|---|---|---|
| `data/train.csv` | 120 titles (30 per source) | `title_id, title, domain, interest` |
| `data/test.csv` | 30 titles (7–8 per source) | `title_id, title, domain` |
| `data/sample_submission.csv` | 30 | `title_id, interest` (3.5 for every title) |

**The key fact about the labels.** Each training title was rated by **one** annotator, but each test label is
the **mean of six** annotators. As a result:

- Training labels are whole numbers spread over the whole scale (mean 3.13, sd 1.68).
- Test labels are averages, so they sit much closer to the middle.
- A single rating is very noisy. A rough estimate: only about 28 % of the variance in one rating is real
  title signal.
- Cross-validation MAE (against single ratings) is much higher than leaderboard MAE (against 6-rater means).
  For example, the constant prediction gets 1.495 in cross-validation but 0.848 on the public board.
  Cross-validation numbers are only useful for **comparing** models.

**Preprocessing** is minimal. The title text is not cleaned, because capital letters, emoji, hashtags and
punctuation are useful cues. The source is one-hot encoded and used in every model.

- Text encoders read `"<domain>: <title>"`, e.g. `"reddit: What is the best advice you ever got?"`.
- LLM prompts use a plain name for the source: "YouTube video", "Reddit post", "podcast episode",
  "news article".

---

## 2. How models are compared

Every notebook uses the same evaluation, so the numbers can be compared across notebooks.

- **Repeated cross-validation:** 5 folds × 10 repeats (`RepeatedKFold(n_splits=5, n_repeats=10,
  random_state=42)`), so every model is fitted 50 times. A single train/dev split of 120 titles would be too
  noisy.
- All fitting (scalers, ridge, SVR, neural networks, fine-tuning) happens inside each fold.
- Predictions are clipped to the range 1–6.
- The score is the mean MAE over the 50 folds. The spread between folds is about ±0.17, so differences
  below about 0.02 are noise.
- The out-of-fold correlation with the labels (`oof_corr`) is also reported. It is less noisy than MAE.
- All random seeds are fixed (42), so every run gives the same numbers.

**Shrinkage** toward the mean (`mean + s × (prediction − mean)`) was tested because the test labels are
compressed. The best `s` was close to 1 and the gain was at noise level, so `s = 1` (no change) is used for
the final files.

---

## 3. Files, in the order they were built

### 3.1 `baseline.ipynb` — simple baselines

**Purpose:** find out how far simple models get with the 120 labels alone.

| Model (5×10 CV) | CV MAE |
|---|---|
| Constant (train mean 3.13) | 1.495 |
| Mean per source | 1.478 |
| Hand-crafted features (length, capitals, `?`, `!`, digits, "you", …) + ridge | 1.494 |
| all-mpnet-base-v2 embeddings + ridge | 1.480 |
| mpnet embeddings + RBF SVR (C = 0.3), shrink s = 0.4 | 1.463 |
| mpnet embeddings + kNN (k = 40, cosine) | 1.469 |
| mpnet embeddings + small PyTorch MLP | 1.72–1.79 (overfits) |

**Outputs:** `submission_constant_mean.csv` (constant), `submission_baseline.csv` (mpnet SVR).

### 3.2 `fetch_webis.py` + `clickbait.ipynb` — external clickbait data

**Idea:** the competition overview names traits such as curiosity gap, exaggeration and direct address.
These are clickbait traits, so a large clickbait dataset might teach the cues.

- **Data:** Webis Clickbait Corpus 2017 (Zenodo record 5530410, CC BY 4.0). It has 40,856 news tweets, each
  rated by 5 crowd workers on a 0–1 clickbait scale.
- **`fetch_webis.py`** downloads only the small JSONL files (posts and labels) from the ~1 GB zip files. It
  uses HTTP range requests, so the images inside the zips are never downloaded.
  Run: `python fetch_webis.py ../external_data/webis_clickbait17`
- **`clickbait.ipynb`**:
  1. Analyses the corpus (surface cues; agreement between raters).
  2. Trains clickbait models on the official Webis train split and tests them on the official test split.
     The best model (mpnet + ridge) reaches Pearson 0.73, close to the best 2017 Clickbait Challenge systems.
  3. Scores the 150 competition titles with the frozen clickbait models.
  4. Tests whether clickbait scores predict interest. They do not: the correlation within each source is
     between −0.26 and +0.14, and the best CV MAE is 1.468–1.473. **Clickbait is not the same as
     interesting**, so this approach was dropped.

**Outputs:** `features/webis_mpnet.npz` (cached mpnet embeddings of the Webis posts, ~128 MB, saves
re-encoding), `submission_clickbait.csv` (never submitted).

### 3.3 `nonllm.ipynb` — a wide range of models without a generative LLM

**Purpose:** check that no conventional model does clearly better than the LLM approach.

| Model (+ source) | CV MAE |
|---|---|
| 24 hand-crafted features × source interactions, ridge (α = 300) | 1.449 |
| bert-base-uncased embeddings, SVR (C = 1) | **1.433** |
| CLIP ViT-H-14 text encoder, kNN | 1.441 |
| mpnet (ridge / SVR / kNN) | 1.47–1.49 |
| Fine-tuned mpnet (all ~110M weights + regression head, 6 epochs) | 1.500 |
| Extra-trees | 1.521 |
| Ensemble of the best model of each family | 1.439 |

Findings:

- Cues mean different things in different sources. Longer podcast titles are rated more interesting
  (Spearman +0.46). Reddit questions are rated less interesting (about −0.4). Hashtags, emoji and brackets
  lower interest on YouTube.
- Fine-tuning a full transformer on ~96 noisy titles per fold only learns the noise.
- The best score (1.433) was picked from 27 models, so it is slightly optimistic.

**Output:** `submission_nonllm.csv` (bert SVR, never submitted).

### 3.4 `llm_rate.py` — local LLM ratings

**Idea:** the test label is an "average reader" judgement, so a language model is asked to act as an average
reader. The model never sees any label, so its scores can be computed once for all 150 titles and used inside
cross-validation without leakage.

- **Model:** Qwen2.5-VL-7B-Instruct, 4-bit (Apache-2.0), run locally with MLX on a Mac (text only).
- **Questions per title:** 2 ratings on the 1–6 scale ("how interesting", "how likely to click") and 5 Yes/No
  traits (curiosity gap, emotional language, exaggeration, direct address, missing details).
- **Readout:** no text is generated. The model runs once per question and the probabilities of the next
  token are read.
  - Rating questions: P("1") … P("6") are rescaled to sum to 1, and the expected value
    `1·p1 + 2·p2 + … + 6·p6` is stored (e.g. 3.74). This keeps the model's uncertainty and is deterministic.
  - Trait questions: `P(Yes) / (P(Yes) + P(No))`.
- **Main prompt:**
  ```
  System: You are an average adult internet user judging content titles. Answer only with what is asked.
  User:   How interesting would a typical reader find this {kind} title?
          Scale: 1 = not interesting at all, 2 = slightly, 3 = somewhat, 4 = fairly, 5 = very, 6 = extremely interesting.
          Title: {title}
          Answer with a single digit from 1 to 6.
  ```
- **Bug fix:** in the first run only 63 % of the probability landed on the digits. The cause was that mlx-vlm
  reused the position information (RoPE) of the first prompt for every later prompt. The script now resets
  it before each call, and the digit mass is ≥ 0.999 for every title. The faulty first run is kept as
  `features/llm_scores_buggy_rope.csv` for reference.
- **Second model:** the same script also ran Qwen3-8B (4-bit, thinking disabled).

```bash
python llm_rate.py
```
```bash
python llm_rate.py mlx-community/Qwen3-8B-4bit features/llm_scores_qwen3_8b.csv
```

**Outputs:** `features/llm_scores.csv` (Qwen2.5), `features/llm_scores_qwen3_8b.csv` (Qwen3), and the exact
prompts in `features/llm_prompts.json` and `features/llm_prompts_qwen3_8b.json`.

Columns: `interest_exp`, `click_exp` (expected rating), `interest_p1..p6`, `click_p1..p6` (probabilities),
`interest_mass`, `click_mass` (share of probability on the digits, a quality check), `trait_<name>`.

### 3.5 `llm_persona.py` — six simulated readers

The same local model rates every title as six different readers (student, working parent, retiree, office
professional, casual browser, sceptical reader), and the six ratings are averaged. The aim was to imitate the
six-annotator test labels. The personas agree strongly with each other (correlation 0.83), so the average adds
little: CV MAE 1.446.

**Outputs:** `features/llm_persona.csv`, `features/llm_persona_prompts.json`.

### 3.6 `llm.ipynb` — turning LLM ratings into predictions

The raw LLM score is on the wrong scale (raw MAE 1.506). A small ridge regression (α = 1, standardised
inputs) on **[LLM rating + 4 source one-hots]** learns one slope and a shift per source.

| LLM features (+ source, ridge) | CV MAE |
|---|---|
| **Interest rating only (Qwen2.5)** | **1.415** |
| + click rating | 1.428 |
| + 5 trait scores | 1.443–1.454 |
| + full 1–6 probability vectors | 1.449+ |
| Six-persona average | 1.446 |
| Raw LLM score, no fit | 1.506 |
| PyTorch MLP (16 hidden units, dropout, 5 seeds) on ratings + traits + source | 1.463 |
| mpnet embeddings + LLM score, SVR | 1.457 |

The single "typical reader" prompt was best; extra prompts mostly added noise. Other ways of fitting the
score (per-source slopes, quadratic term, isotonic, quantile, Bayesian ridge, Gaussian process, gradient
boosting, blending with BERT) did not beat the plain ridge in a reliable way.

Qwen3-8B ranked titles slightly better but was over-confident (CV 1.439). Averaging the z-scored Qwen2.5 and
Qwen3 ratings gave CV 1.414.

**Outputs:** `submission_llm_linear.csv` (ridge on Qwen2.5 rating), `submission_llm_mlp.csv` (MLP).

### 3.7 `claude_rate.py` — Claude Opus 5.5 ratings (best model)

Every step up in model quality had helped, so a much larger hosted model was tried.

- **Model:** `claude-opus-5-5` through the official Anthropic Python SDK. The API key is read from the
  environment variable `ANTHROPIC_API_KEY` and is never stored in the project.
- **Readout:** the API does not return token probabilities. Instead, Claude is asked for a probability for
  each score 1–6, and a JSON schema forces the answer into the form `{p1, …, p6}`. The six numbers are
  normalised and turned into the expected rating `claude_exp`.
- **Prompt:** the same "typical reader" question as the local model, plus: *"Different readers disagree, so
  give the probability that a randomly chosen typical reader would pick each score."*
- **Settings:** effort "low"; 4 requests in parallel; each result is written to the CSV at once, so a rerun
  only scores missing titles and nothing is paid twice. Total cost was about $0.46.
- The API has no random seed, so results are reproduced from the saved CSV rather than by calling the API
  again.

```bash
ANTHROPIC_API_KEY=... python claude_rate.py
```

**Outputs:** `features/claude_scores.csv` (`claude_exp`, `claude_p1..p6`, `served_by`),
`features/claude_prompts.json`.

| Rating (+ source, ridge α = 1) | Spearman with labels | CV MAE |
|---|---|---|
| Qwen2.5-7B | 0.19 | 1.415 |
| **Claude Opus 5.5** | **0.38** | **1.378** |
| Claude + Qwen2.5 averaged | 0.32 | 1.381 |

Claude's rating correlates twice as strongly with the labels as the local models, and it is positive in every
source, including news articles, where the local models were at or below zero.

**Building `submission_claude.csv`.** The same ridge as in `llm.ipynb` is fitted on `claude_exp` plus the
source one-hots. This code reproduces the file exactly:

```python
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
scores = pd.read_csv("features/claude_scores.csv")[["title_id", "claude_exp"]]
train, test = train.merge(scores), test.merge(scores)

def features(d):
    dom = pd.get_dummies(d.domain).reindex(columns=["article", "podcast", "reddit", "youtube"], fill_value=0)
    return np.column_stack([d.claude_exp, dom.values.astype(float)])

model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(features(train), train.interest)
pred = np.clip(model.predict(features(test)), 1, 6)
pd.DataFrame({"title_id": test.title_id, "interest": pred.round(3)}).to_csv("submission_claude.csv", index=False)
```

---

## 4. Results

CV MAE uses single-annotator train labels; public MAE uses 6-annotator means, so the two are on different
scales.

| Submission file | Approach | CV MAE | Public MAE |
|---|---|---|---|
| `submission_constant_mean.csv` | Constant: train mean 3.133 | 1.495 | 0.848 |
| `submission_baseline.csv` | mpnet embeddings + source, RBF SVR (C = 0.3), shrink 0.4 | 1.463 | 0.828 |
| `submission_llm_linear.csv` | Qwen2.5-7B rating + source, ridge | 1.415 | 0.724 |
| `submission_claude.csv` | Claude Opus 5.5 rating + source, ridge | **1.378** | **0.629** |

Kaggle reference scores: mean benchmark 0.848; `sample_submission.csv` (3.5 everywhere) 0.921.

Prepared but not submitted:

| File | Approach | CV MAE |
|---|---|---|
| `submission_llm_mlp.csv` | PyTorch MLP on LLM features | 1.463 |
| `submission_nonllm.csv` | bert-base embeddings + SVR | 1.433 |
| `submission_llm2_avg.csv` | Average of Qwen2.5 and Qwen3 ratings, ridge | 1.414 |
| `submission_sklearn_handcrafted.csv` | Hand-crafted features × source, ridge (α = 300) | 1.449 |
| `submission_llm_bert_blend.csv` | Blend of LLM ridge and bert SVR | 1.411–1.414 |
| `submission_clickbait.csv` | Clickbait-score features | 1.468–1.473 |

**Final selections:** `submission_claude.csv` (best CV and best public score) and `submission_llm_linear.csv`
(best model that runs fully on a local machine).

---

## 5. What worked and why

1. **The labels are the bottleneck.** Every model trained only on the 120 single-rater labels (TF-IDF,
   hand-crafted features, embeddings, fine-tuned transformers, trees, MLPs) landed between 1.43 and 1.50,
   barely better than the constant (1.495).
2. **A zero-shot LLM supplies the "average reader" view.** It was never trained on the noisy labels and
   brings a general sense of what people find interesting, which is what a 6-rater mean measures. This is
   why the LLM runs improved the public score much more than the CV score.
3. **Model quality mattered more than modelling tricks.** Public MAE improved with each stronger model:
   sentence embeddings 0.828 → 7B local LLM 0.724 → Claude Opus 5.5 0.629. Extra prompts, personas, traits,
   fancier fitting and ensembles changed CV MAE by less than ±0.01.
4. **Keep the learned part tiny.** The best use of the 120 labels was to learn only the scale and a shift per
   source (5 numbers). More flexible models fitted noise.
5. **Readouts that keep uncertainty help.** The expected value of a probability distribution gives smoother
   scores than a single sampled answer.

What did not work: clickbait transfer (a different concept from interest), fine-tuning (learns single-rater
noise), personas (the same model gives highly correlated answers). The public board has only ~15 titles
(standard error about ±0.12), so final choices were based on CV, not on the public board.

---

## 6. Folder map

```
data/                      competition data (train, test, sample submission)
features/                  saved model outputs, reused by the notebooks
  llm_scores.csv             Qwen2.5 ratings + traits            (llm_rate.py)
  llm_scores_qwen3_8b.csv    Qwen3-8B ratings + traits           (llm_rate.py)
  llm_scores_buggy_rope.csv  first Qwen2.5 run before the RoPE fix, kept for reference
  llm_persona.csv            six persona ratings                 (llm_persona.py)
  claude_scores.csv          Claude Opus 5.5 probabilities        (claude_rate.py)
  llm_combined.csv           all LLM scores merged per title, used for comparisons
  *_prompts*.json            exact model names and prompt texts for each run
  webis_mpnet.npz            cached mpnet embeddings of the Webis corpus (clickbait.ipynb)
../external_data/webis_clickbait17/   Webis JSONL files (fetch_webis.py)
baseline.ipynb             baselines (constant, hand-crafted, mpnet, MLP)
clickbait.ipynb            clickbait transfer experiment
nonllm.ipynb               wider set of non-LLM models, incl. fine-tuned mpnet
llm.ipynb                  ridge / MLP on the LLM ratings
fetch_webis.py, llm_rate.py, llm_persona.py, claude_rate.py   scripts that create the features above
submission_*.csv           prediction files (see section 4)
writeup.md, writeup_full.md (+ .docx)   written report
```

## 7. Software and licences

- Python 3.13 with numpy, pandas, scipy, scikit-learn, matplotlib.
- PyTorch and Hugging Face `transformers` (mpnet, bert-base-uncased, fine-tuning); `open_clip_torch`
  (CLIP ViT-H-14, `laion2b_s32b_b79k`).
- `mlx`, `mlx-lm`, `mlx-vlm` for the local LLMs (Apple Silicon Mac).
- `anthropic` (official SDK) for `claude_rate.py`, with `ANTHROPIC_API_KEY` set in the environment.

All external models and data have open licences: Qwen2.5-VL and Qwen3 (Apache-2.0), mpnet (Apache-2.0),
bert-base-uncased (Apache-2.0), OpenCLIP (MIT), Webis Clickbait 2017 (CC BY 4.0).
