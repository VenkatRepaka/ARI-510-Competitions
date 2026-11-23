"""Builds nonllm.ipynb: a wider comparison of models that use NO large language model
(run: python make_nonllm_nb.py && jupyter nbconvert --to notebook --execute --inplace nonllm.ipynb)."""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Models without an LLM — wider search

`models.ipynb` tried a few non-LLM models (regex features, mpnet / BERT embeddings) and none beat the
simple reference "predict insufficient for every question" (F1 0.51). This notebook tries many more
options, grouped by the kind of input they use:

| group | input | models tried |
|---|---|---|
| A. one number | a single feature (length, share of common words, technical tokens) | logistic regression = a learned cut-off |
| B. hand-made features | regex cues (`feats.handcrafted`) and writing-style counts (`nonllm_feats.text_stats`) | logistic regression (L2, L1), naive Bayes, k-nearest neighbours, LDA, SVM, random forest, extra trees, gradient boosting |
| C. words | TF-IDF of words or character n-grams, word counts | logistic regression, complement naive Bayes |
| D. sentence embeddings | mpnet / BERT vectors of the full text, the title only, the body only, or title and body side by side | logistic regression, SVM, k-nearest neighbours, LDA, PCA + logistic regression |
| E. combinations | soft vote (average probability) of models from different groups | |

Every model is scored the same way as before (`feats.cv_eval`): 5-fold stratified CV repeated 5 times,
F1 of class 1, with the decision threshold chosen inside each training part only.
`AUC` is computed from the out-of-fold probabilities of the first repeat (0.5 = random ordering).
Text-to-number steps that learn from data (TF-IDF vocabulary, scaling, PCA) sit inside the pipeline,
so they are refit on each training part; the sentence encoders are pretrained and never see labels.
""")

code("""
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from feats import SEED, handcrafted, cv_eval
from nonllm_feats import text_stats, embed_parts

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y, n = train.sufficiency.values, len(train)
texts = np.concatenate([train.text.values, test.text.values])   # train first, then test (same order as the cached embeddings)

H = handcrafted(texts); S = text_stats(texts)
HS = pd.concat([H, S], axis=1)
E = {k: np.load(f"features/emb_{k}.npy") for k in ["mpnet", "bert"]}
for tag, name in [("mpnet", "sentence-transformers/all-mpnet-base-v2"), ("bert", "bert-base-uncased")]:
    E[f"{tag}_title"], E[f"{tag}_body"] = embed_parts(texts, name, tag)
    E[f"{tag}_title+body"] = np.hstack([E[f"{tag}_title"], E[f"{tag}_body"]])
print("hand-made features:", H.shape[1], "+ text stats:", S.shape[1], "| embeddings:", {k: v.shape[1] for k, v in E.items()})
print("reference F1 (predict all insufficient):", round(f1_score(y, np.ones(n)), 3))
""")

code("""
# Model zoo. Every model ends in predict_proba so the same CV routine can tune its threshold.
def lr(C=1.0, penalty="l2"):
    return LogisticRegression(C=C, penalty=penalty, solver="liblinear", class_weight="balanced", random_state=SEED)
def scaled(m): return make_pipeline(StandardScaler(), m)

TABULAR = {
    "LR C=0.1":            scaled(lr(0.1)),
    "LR L1 C=0.3":         scaled(lr(0.3, "l1")),          # L1 keeps only a few features
    "Gaussian naive Bayes": scaled(GaussianNB()),
    "kNN k=7":             scaled(KNeighborsClassifier(7, weights="distance")),
    "LDA (shrinkage)":     LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "SVM RBF":             scaled(SVC(C=1, class_weight="balanced", probability=True, random_state=SEED)),
    "random forest":       RandomForestClassifier(300, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
    "extra trees":         ExtraTreesClassifier(300, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
    "gradient boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED),
}
EMBEDDING = {
    "LR C=0.1":        lr(0.1),
    "LR C=1":          lr(1.0),
    "SVM RBF":         SVC(C=1, class_weight="balanced", probability=True, random_state=SEED),
    "kNN k=7 cosine":  KNeighborsClassifier(7, metric="cosine", weights="distance"),
    "LDA (shrinkage)": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "PCA 10 + LR":     make_pipeline(PCA(10, random_state=SEED), StandardScaler(), lr(0.1)),
}

results, OOF = [], {}
def run(group, inputs, model_name, model, X):
    t0 = time.time()
    r = cv_eval(model, X, y)
    OOF[(group, inputs, model_name)] = r["oof"]
    results.append({"group": group, "input": inputs, "model": model_name, "F1 CV tuned": r["f1_tuned"],
                    "std": r["f1_tuned_std"], "F1 @0.5": r["f1@0.5"], "AUC": roc_auc_score(y, r["oof"]),
                    "sec": time.time() - t0})
""")

md("## A — one number with a learned cut-off")
code("""
for col, src in [("log_words", H), ("log_chars", S), ("share_common_words", S), ("n_tech_tokens", S),
                 ("type_token_ratio", S), ("first_person_detail", H)]:
    run("A one number", col, "LR", scaled(lr(1.0)), src[[col]].values[:n])
pd.DataFrame(results).query("group == 'A one number'").round(3)
""")

md("## B — hand-made features")
code("""
for inputs, X in [("regex cues (15)", H.values[:n]), ("text stats (16)", S.values[:n]), ("regex + stats (31)", HS.values[:n])]:
    for name, m in TABULAR.items():
        run("B hand-made", inputs, name, m, X)
pd.DataFrame(results).query("group == 'B hand-made'").sort_values("F1 CV tuned", ascending=False).round(3)
""")

md("## C — words (bag of words / character n-grams)")
code("""
T = texts[:n]
WORDS = {
    "TF-IDF words 1-2 + LR":        make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), lr(1.0)),
    "TF-IDF words 1-2 + LR C=10":   make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), lr(10.0)),
    "TF-IDF char 2-5 + LR":         make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), lr(1.0)),
    "word counts + complement NB":  make_pipeline(CountVectorizer(min_df=2), ComplementNB()),
    "char counts + complement NB":  make_pipeline(CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2), ComplementNB()),
    # function words only: HOW something is asked, independent of the product (Outlook, iPhone ...)
    "common-word counts + LR":      make_pipeline(CountVectorizer(vocabulary=sorted(__import__("nonllm_feats").COMMON)),
                                                  FunctionTransformer(lambda X: np.log1p(X.toarray())), StandardScaler(), lr(0.1)),
}
for name, m in WORDS.items():
    run("C words", "text", name, m, T)
pd.DataFrame(results).query("group == 'C words'").sort_values("F1 CV tuned", ascending=False).round(3)
""")

md("## D — sentence embeddings")
code("""
for inputs in ["mpnet", "mpnet_title", "mpnet_body", "mpnet_title+body", "bert", "bert_title", "bert_body", "bert_title+body"]:
    for name, m in EMBEDDING.items():
        run("D embeddings", inputs, name, m, E[inputs][:n])
pd.DataFrame(results).query("group == 'D embeddings'").sort_values("F1 CV tuned", ascending=False).head(15).round(3)
""")

md("""
## E — combinations (soft vote)

Each member sees only its own columns of one big matrix (hand-made features | BERT | mpnet body), and the
vote averages their probabilities. Members are fixed in advance (one simple model per input type),
not picked from the tables above, so the score is not inflated by choosing the winners.
""")
code("""
nh, nb, nm = HS.shape[1], E["bert"].shape[1], E["mpnet_body"].shape[1]
X_all = np.hstack([HS.values, E["bert"], E["mpnet_body"]])[:n]
def cols(a, b): return ColumnTransformer([("keep", "passthrough", list(range(a, b)))])
m_hand = make_pipeline(cols(0, nh), StandardScaler(), lr(0.1))
m_bert = make_pipeline(cols(nh, nh + nb), lr(0.1))
m_mpb = make_pipeline(cols(nh + nb, nh + nb + nm), lr(0.1))
m_rf = make_pipeline(cols(0, nh), RandomForestClassifier(300, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=SEED))
for name, members in [("hand-made LR + BERT LR", [m_hand, m_bert]),
                      ("hand-made LR + mpnet body LR", [m_hand, m_mpb]),
                      ("hand-made LR + BERT LR + mpnet body LR", [m_hand, m_bert, m_mpb]),
                      ("hand-made LR + hand-made RF + BERT LR + mpnet body LR", [m_hand, m_rf, m_bert, m_mpb])]:
    run("E vote", "mixed", name, VotingClassifier([(f"m{i}", m) for i, m in enumerate(members)], voting="soft"), X_all)
pd.DataFrame(results).query("group == 'E vote'").round(3)
""")

md("## All non-LLM results")
code("""
res = pd.DataFrame(results).sort_values("F1 CV tuned", ascending=False)
res["beats all-1 (0.51)"] = res["F1 CV tuned"] > 0.512
res.to_csv("logs/nonllm_results.csv", index=False)
print(f"{len(res)} models tried, {res['beats all-1 (0.51)'].sum()} beat the all-1 reference; "
      f"{(res.AUC >= 0.6).sum()} have AUC >= 0.6")
res.head(20).round(3)
""")

code("""
# Best model per group, next to the reference lines
best = res.loc[res.groupby("group")["F1 CV tuned"].idxmax()].sort_values("F1 CV tuned")
fig, ax = plt.subplots(figsize=(7.5, 3))
lab = best.group + ": " + best.input + " / " + best.model
ax.barh(range(len(best)), best["F1 CV tuned"], xerr=best["std"], color="#2a78d6", height=0.6,
        error_kw={"ecolor": "#52514e", "lw": 1})
ax.axvline(0.512, color="#52514e", ls="--", lw=1); ax.text(0.515, len(best) - 0.4, "predict all insufficient", fontsize=8, color="#52514e")
ax.axvline(0.611, color="#eb6834", ls=":", lw=1); ax.text(0.614, -0.45, "LLM checklist (run2)", fontsize=8, color="#eb6834")
ax.set_yticks(range(len(best))); ax.set_yticklabels(lab, fontsize=8)
ax.set_xlim(0, 0.8); ax.set_xlabel("F1 of 'insufficient', 5x5 CV (bar = mean, line = std over folds)")
ax.spines[["top", "right"]].set_visible(False); ax.set_title("Best non-LLM model per group", loc="left")
plt.tight_layout(); plt.show()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("nonllm.ipynb", "w"), indent=1)
