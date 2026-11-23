"""Shared feature builders and the cross-validation routine used by models.ipynb.

HANDCRAFTED FEATURES
--------------------
Reading the 64 labelled questions, "insufficient" questions tend to miss things a support person needs:
the exact error text ("threw this error" with no message), the device / version ("my iPhone ios"),
a concrete symptom, or a file/data the reader cannot see ("data file enclosed"). "Sufficient" ones
are usually longer, name versions, quote messages and list what was already tried.
`handcrafted()` turns these cues into numbers with simple regular expressions (one row per question).

EMBEDDINGS
----------
`embed()` turns each question into one vector with a pretrained sentence encoder (mean of the
token vectors, then length 1). Only models already cached on this machine are used.

CROSS-VALIDATION WITH A TUNED THRESHOLD
---------------------------------------
The metric is F1 of class 1 ("insufficient"), and only 22 of 64 questions are class 1, so the
default cut-off 0.5 on the predicted probability is rarely the best one. `cv_eval()`:
  1. splits the 64 questions into 5 folds (stratified = same class ratio in each fold), repeated 5 times;
  2. inside each training part, runs a second (inner) 4-fold CV to get honest probabilities and picks
     the threshold that maximises F1 there;
  3. applies the model and that threshold to the held-out fold.
So the threshold is never chosen on the questions it is scored on.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

SEED = 42


def split_title(text):
    """Split 'Title: ... Question: ...' into (title, body)."""
    m = re.match(r"\s*Title:\s*(.*?)\s*Question:\s*(.*)", text, flags=re.S)
    return (m.group(1), m.group(2)) if m else ("", text)


def handcrafted(texts):
    """Regex cues for 'is the needed information there?'. Returns a DataFrame (one row per text)."""
    rows = []
    for t in texts:
        title, body = split_title(t)
        b = body.lower()
        words = body.split()
        rows.append({
            "log_words": np.log1p(len(words)),                       # longer posts usually carry more detail
            "log_title_words": np.log1p(len(title.split())),
            "n_sentences": len(re.findall(r"[.!?](\s|$)", body)) + 1,
            "n_questions": body.count("?"),
            # version numbers like 26.4.2, 2016, build numbers, "version"
            "has_version": int(bool(re.search(r"\b\d+\.\d+|\bversion\b|\bbuild\b|\b20[0-2]\d\b", b))),
            # operating system / platform named
            "has_os": int(bool(re.search(r"windows ?\d+|win ?1[01]|mac ?os|catalina|big sur|ios ?\d+|android|macbook|surface", b))),
            # device model ("iphone 15", "18 pro max")
            "has_device": int(bool(re.search(r"iphone ?\d+|\bpro max\b|ipad|homepod|airpods|\bmac\b", b))),
            # an error is mentioned ...
            "mentions_error": int(bool(re.search(r"\berror|fail|crash|not working|doesn'?t work|won'?t", b))),
            # ... and something is quoted (an exact message)
            "has_quote": int(bool(re.search(r"[\"“”']{1}[^\"“”']{6,}[\"“”']", body))),
            # what the user already tried
            "tried": int(bool(re.search(r"\btried|\bhave (checked|tested|restarted|reinstalled)|restart|reinstall|repair", b))),
            # refers to something not visible in the text
            "attachment": int(bool(re.search(r"enclosed|attached|attachment|screenshot|see below|as below|picture|image above", b))),
            # how-to / "is it possible" style questions
            "howto": int(bool(re.search(r"^(how|is there|is it possible|can i|can we|what|why)\b|\bhow (do|to|can)\b", b))),
            "retitled": int("re-titled by moderator" in b),
            # steps / numbered lists / settings paths
            "steps": int(bool(re.search(r"\b\d\.\s|->|→|>\s?\w|settings", b))),
            "first_person_detail": len(re.findall(r"\bi (have|am|tried|checked|use|installed|switched)\b", b)),
        })
    return pd.DataFrame(rows)


def embed(texts, model_name="sentence-transformers/all-mpnet-base-v2", device="cpu", max_length=512, batch_size=8):
    """Mean-pooled, L2-normalised sentence embeddings from a Hugging Face encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
            h = enc(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.vstack(out)


def best_threshold(y, p):
    """Threshold on probability p that maximises F1 of class 1 (ties -> the middle of the best range)."""
    grid = np.linspace(0.05, 0.95, 91)
    f = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    best = grid[f == f.max()]
    return float(np.median(best))


def oof_proba(model, X, y, n_splits=4, seed=SEED):
    """Out-of-fold P(class 1) for every row (each row predicted by a model that did not see it)."""
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        p[va] = clone(model).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return p


def cv_eval(model, X, y, n_repeats=5, tune_threshold=True, seed=SEED):
    """5x5 stratified CV. Returns dict with F1 (tuned and at 0.5), its std, and repeat-0 OOF probabilities."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    f_tuned, f_05, oof = [], [], np.zeros(len(y))
    for k, (tr, va) in enumerate(cv.split(X, y)):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        if k < 5:
            oof[va] = p
        f_05.append(f1_score(y[va], p >= 0.5, zero_division=0))
        if tune_threshold:
            t = best_threshold(y[tr], oof_proba(model, X[tr], y[tr], seed=seed + k))
            f_tuned.append(f1_score(y[va], p >= t, zero_division=0))
    res = {"f1@0.5": np.mean(f_05), "f1@0.5_std": np.std(f_05), "oof": oof}
    if tune_threshold:
        res.update({"f1_tuned": np.mean(f_tuned), "f1_tuned_std": np.std(f_tuned)})
    return res
