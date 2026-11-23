"""Builds models.ipynb (run: python make_models_nb.py && jupyter nbconvert --execute --inplace models.ipynb).

Keeping the notebook as generated code makes it easy to rebuild it from scratch with fixed seeds.
"""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Troubleshooting sufficiency — model comparison

Goal: predict `1` (insufficient information) vs `0` (sufficient) for tech-support questions.
Metric: **F1 of class 1** (the public-leaderboard scores 6/13, 16/27, 9/17 fit binary F1 on ~25 public rows).

Every model is scored with the same 5-fold stratified CV repeated 5 times (`feats.cv_eval`), with the
decision threshold chosen inside each training part (inner CV), never on the held-out fold.

Reference points:
* predicting `1` for everything gives F1 ≈ 0.51 (22 of 64 questions are class 1) — any model must beat this;
* run1 (mpnet embeddings + LR, threshold 0.5) — CV 0.454, public 0.46153.

Feature families:
1. **handcrafted** regex cues (`feats.handcrafted`): length, versions/OS/device named, error mentioned/quoted, tried steps, attachments…
2. **sentence embeddings** of the full text from cached encoders (mpnet, BERT-base);
3. **LLM checklist** (`llm_judge.py`): a local LLM answers Yes/No questions ("enough information to start troubleshooting?",
   "product and version named?", …); the soft P(Yes) values are the features. Prompts are saved in `features/llm_prompts_*.json`.
""")

code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, random, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from feats import SEED, handcrafted, cv_eval, best_threshold, oof_proba

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y, n = train.sufficiency.values, len(train)
all_text = np.concatenate([train.text.values, test.text.values])
print(train.shape, test.shape, "class-1 share", y.mean().round(3))
print("F1 of predicting all 1:", round(f1_score(y, np.ones_like(y)), 3))
""")

md("## Features")

code("""
H = handcrafted(all_text)
E = {k: np.load(f"features/emb_{k}.npy") for k in ["mpnet", "bert"]}      # computed with feats.embed (see exp scripts)

L = {}
for tag in ["q25", "q3"]:
    try:
        d = pd.read_csv(f"features/llm_{tag}.csv").set_index("id").loc[np.concatenate([train.id, test.id])]
        L[tag] = d[[c for c in d.columns if not c.endswith("_mass")]]
        print(tag, "mean Yes/No mass", d.filter(like="_mass").mean().round(2).to_dict())
    except FileNotFoundError:
        print("missing", tag)

# How well does each single LLM answer separate the classes on train? (AUC 0.5 = no signal)
auc = {f"{tag}:{c}": roc_auc_score(y, L[tag][c].values[:n]) for tag in L for c in L[tag]}
pd.Series(auc).sort_values().round(3)
""")

md("## Cross-validated comparison")

code("""
def LR(C, scale=True):
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
    return make_pipeline(StandardScaler(), m) if scale else m

results, oofs = {}, {}
def run(name, model, X):
    r = cv_eval(model, X[:n], y)
    results[name] = {k: v for k, v in r.items() if k != "oof"}
    oofs[name] = r["oof"]
    print(f"{name:45s} F1 tuned {r['f1_tuned']:.3f} ± {r['f1_tuned_std']:.3f}   F1@0.5 {r['f1@0.5']:.3f}")

Hv = H.values.astype(float)
run("handcrafted LR C=0.03", LR(0.03), Hv)
run("mpnet LR C=1", LR(1, scale=False), E["mpnet"])
run("bert LR C=1", LR(1, scale=False), E["bert"])
run("bert SVM C=10", SVC(C=10, probability=True, class_weight="balanced", random_state=SEED), E["bert"])

for tag in L:
    Lv = L[tag].values
    run(f"LLM {tag}: 'sufficient' only, LR", LR(1), Lv[:, [0]])
    for C in (0.03, 0.3):
        run(f"LLM {tag}: checklist LR C={C}", LR(C), Lv)
    run(f"LLM {tag}: checklist + handcrafted LR C=0.03", LR(0.03), np.hstack([Lv, Hv]))
if len(L) == 2:
    both = np.hstack([L["q25"].values, L["q3"].values])
    run("LLM q25+q3 checklist LR C=0.03", LR(0.03), both)
""")

md("""
### Small neural network

A one-hidden-layer MLP (16 units, strong L2) on the LLM checklist + handcrafted features.
""")

code("""
best_tag = max(L, key=lambda t: results[f"LLM {t}: checklist LR C=0.03"]["f1_tuned"]) if L else None
if best_tag:
    X_nn = np.hstack([L[best_tag].values, Hv])
    run(f"MLP 16 on LLM {best_tag} + handcrafted",
        make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=1.0, max_iter=3000, random_state=SEED)), X_nn)
""")

md("## Results")

code("""
res = pd.DataFrame(results).T.sort_values("f1_tuned", ascending=False)
res.round(3)
""")

md("""
## Submission

Fit the chosen model on all 64 questions, choose the threshold from out-of-fold probabilities on the
training set (same rule as inside CV), and predict the 50 test questions.
""")

code("""
BEST = res.index[0]          # <- best by CV; change by hand to submit another row
print("chosen:", BEST, results[BEST])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("models.ipynb", "w"), indent=1)
