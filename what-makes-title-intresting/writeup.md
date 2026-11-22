# What Makes Title Interesting — Approaches

Competition: https://www.kaggle.com/competitions/what-makes-title-intresting

Task: predict how interesting a title is, on a scale from 1 to 6. Metric: Mean Absolute Error (lower is better).

## 2. Approaches

### 2.1 Data and evaluation

The training set has 120 titles, 30 from each of four sources: YouTube, Reddit, podcasts and news articles. The
test set has 30 titles. I did not clean the titles, because capital letters, emoji and punctuation can change
how interesting a title looks. The source of each title was one-hot encoded and used in every model.

With only 120 titles, one train/dev split is not reliable. So I evaluated every model with 5-fold
cross-validation repeated 10 times and reported the average MAE. All fitting was done inside each fold, and all
seeds were fixed to 42. Differences smaller than about 0.02 were treated as noise.

### 2.2 Baselines and models without an LLM

I started with simple models: the training mean (dev MAE 1.495), the mean per source (1.478), and mpnet sentence
embeddings with ridge, SVR, kNN and a small neural network. The best was the embedding SVR (1.463), which was my
second run.

Then I tried many other models without an LLM: hand-made features (length, capital letters, question marks,
numbers, emoji, "you"), TF-IDF, BERT and CLIP embeddings, random forest, gradient boosting and a fine-tuned mpnet.
The best was BERT with SVR (1.433). Fine-tuning mpnet overfitted (1.500), because about 96 titles per fold are
not enough. The features also showed that the same cue means different things per source: longer podcast titles
were more interesting, but questions on Reddit and emoji on YouTube were less interesting.

I also trained clickbait models on the Webis Clickbait 2017 dataset (CC BY 4.0, about 41,000 tweets). They
worked well on clickbait (correlation 0.72), but the clickbait score did not predict interest. Clickbait is not
the same as interesting, so I dropped it.

The main lesson was that 120 labels alone are not enough to learn what makes a title interesting.

### 2.3 Local LLM (zero-shot)

The score we predict is the opinion of an average reader, so I asked an LLM to act as one. I ran
Qwen2.5-VL-7B-Instruct (Apache-2.0) locally with the MLX library. The model never saw any labels. Prompt:

> How interesting would a typical reader find this {kind} title?
> Scale: 1 = not interesting at all, 2 = slightly, 3 = somewhat, 4 = fairly, 5 = very, 6 = extremely interesting.
> Title: {title}
> Answer with a single digit from 1 to 6.

I read the probabilities of the digits 1 to 6 and calculated the expected rating. Then a small ridge model (LLM
score + source, only five learned numbers) put the score on the right scale. Dev MAE was 1.415, and the public
score improved from 0.83 to 0.724 (third run).

Other ideas did not beat this simple prompt: a "how likely are you to click" prompt, five yes/no questions about
the title style, six reader personas, and a second model (Qwen3-8B). I also found and fixed a bug: the MLX
library reused position information from the first prompt, so I reset it before every call.

### 2.4 Pairwise comparisons

I also asked the LLM "Which of these two titles is more interesting? A or B", for at least 12 random pairs per
title, in both orders. The answers were turned into one score per title. This gave the best local dev score
(1.393), but a worse public score (0.812). The public test set is very small, so I decided pairwise was not
clearly better.

### 2.5 Large LLM: Claude Opus 5.5

Better models always gave better results, so I tried Claude Opus 5.5 through the API. The code calls the API
once per title. The API does not give token probabilities, so the prompt asked for a probability for each score
from 1 to 6 in a fixed JSON format:

> How interesting would a typical reader find this {kind} title? Scale: 1 = not interesting at all … 6 =
> extremely interesting. Title: {title}. Different readers disagree, so give the probability that a randomly
> chosen typical reader would pick each score (p1..p6, summing to 1).

I calculated the expected rating and used the same small ridge model. Claude's ratings were twice as strongly
related to the labels as the local model's (correlation 0.38 vs 0.19). The cost was about $0.46. The API has no
seed, so the saved ratings are used to reproduce the run.

### 2.6 Neural networks

I trained four PyTorch models: an MLP on mpnet embeddings (overfitted), an MLP on the clickbait data, a
fine-tuned mpnet, and an MLP on the LLM scores (dev MAE 1.463, better than the baselines).

### 2.7 Summary

| Approach | Dev MAE | Public MAE |
|---|---|---|
| Training mean | 1.495 | 0.848 |
| mpnet embeddings + SVR | 1.463 | 0.828 |
| Qwen2.5-7B rating + ridge | 1.415 | 0.724 |
| Qwen2.5-7B pairwise + ridge | 1.393 | 0.812 |
| **Claude Opus 5.5 rating + ridge** | **1.378** | **0.629** |

The best results came from asking an LLM to judge each title like an average reader and keeping the learned
part very small. A larger LLM gave a clearly better result.
