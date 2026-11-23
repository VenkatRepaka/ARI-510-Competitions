"""Write the LLM-based submission files (reproducible: fixed seed, features read from features/).

run2  submission_llm_checklist.csv
      Logistic regression (C=0.03, balanced class weights, standardised inputs) on the 8 Qwen2.5 checklist
      scores from llm_judge.py, cut-off 0.5.   Dev (5x5 CV): F1 0.611 @0.5 (0.589 with tuned threshold).
run3  submission_llm_wording_mean.csv
      Average of the 8 zero-shot prompt wordings v1-v8 from prompt_variants.py (Qwen2.5), then a 1-feature
      logistic regression to set the cut-off.  Dev: F1 0.602 (tuned) / 0.569 @0.5.
      Only 2 numbers are learned from the labels, so it is the least likely to be overfitted.
run4  submission_tfidf.csv   (no LLM)
      TF-IDF of words and word pairs (min_df=2, sublinear tf) + logistic regression (C=10, balanced),
      cut-off chosen on out-of-fold training probabilities. Best of the 91 non-LLM models in nonllm.ipynb.
      Dev: F1 0.536 (tuned), about the same as predicting every question insufficient (0.512).

Usage: python make_submission.py
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feats import SEED, best_threshold, oof_proba

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y, n = train.sufficiency.values, len(train)
ids = np.concatenate([train.id, test.id])


def load(path):
    """LLM scores in train-then-test order, without the *_mass quality-check columns."""
    d = pd.read_csv(path).set_index("id").loc[ids]
    return d[[c for c in d.columns if not c.endswith("_mass")]]


def lr(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED))


def write(name, pred):
    sub = pd.DataFrame({"id": test.id, "sufficiency": pred.astype(int)})
    sub.to_csv(name, index=False)
    print(f"wrote {name}: {sub.sufficiency.sum()} of {len(sub)} predicted insufficient")


# run2: checklist features, cut-off 0.5
X = load("features/llm_q25.csv").values
write("submission_llm_checklist.csv", lr(0.03).fit(X[:n], y).predict_proba(X[n:])[:, 1] >= 0.5)

# run3: mean of 8 wordings; cut-off chosen on out-of-fold training probabilities (same rule as in CV)
V = load("features/prompt_variants_q25.csv")
x = V[[c for c in V.columns if c != "v9_fewshot"]].mean(1).values.reshape(-1, 1)
m = lr(0.03)
t = best_threshold(y, oof_proba(m, x[:n], y))
print(f"run3 threshold {t:.2f}")
write("submission_llm_wording_mean.csv", m.fit(x[:n], y).predict_proba(x[n:])[:, 1] >= t)

# run4 (no LLM): TF-IDF words 1-2 + LR C=10, same settings as in nonllm.ipynb; vocabulary learned on training text only
m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                  LogisticRegression(C=10.0, solver="liblinear", class_weight="balanced", random_state=SEED))
T = train.text.values
t = best_threshold(y, oof_proba(m, T, y))
print(f"run4 threshold {t:.2f}")
write("submission_tfidf.csv", m.fit(T, y).predict_proba(test.text.values)[:, 1] >= t)
