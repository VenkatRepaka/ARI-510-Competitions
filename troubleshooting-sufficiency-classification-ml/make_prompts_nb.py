"""Builds prompts.ipynb, the prompt-wording comparison
(run: python make_prompts_nb.py && jupyter nbconvert --to notebook --execute --inplace prompts.ipynb)."""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Which prompt wording works? — LLM prompt comparison

The same question ("does this support question contain sufficient information?") is asked in 9 different
wordings (`prompt_variants.py`), with two local LLMs (Qwen2.5-VL-7B and Qwen3-8B, both 4-bit, run with MLX).
Every variant gives one number per question: a score in [0, 1] where **higher = more likely insufficient**
(class 1), read from the first-token probabilities (no text is generated, so results are deterministic).

The 8 checklist prompts from `llm_judge.py` (product/version named? exact error quoted? ...) are compared
the same way in the second half.

How each prompt is judged, on the 64 labelled training questions:

| column | meaning | needs training? |
|---|---|---|
| `mass` | share of probability on the allowed answers (Yes/No or 1-5); low = the model ignored the format | no |
| `AUC` | does the score rank insufficient questions above sufficient ones? 0.5 = random, 1 = perfect | no |
| `F1 zero-shot @0.5` | F1 of class 1 if we simply predict "insufficient" when score >= 0.5 | no |
| `share pred 1` | fraction of questions predicted insufficient at 0.5 (true share is 0.34) | no |
| `F1 CV tuned` | 5x5 CV: logistic regression on this one score, threshold chosen inside each training part | yes (2 numbers + threshold) |

Reference: predicting "insufficient" for every question gives F1 0.51 — a prompt that does not beat this is not useful.
The few-shot variant shows 4 training questions as examples, so those 4 are left out of its evaluation.
""")

code("""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from feats import SEED, cv_eval

train = pd.read_csv("data/train.csv")
y_all = train.set_index("id").sufficiency
MODELS = {"q25": "Qwen2.5-VL-7B", "q3": "Qwen3-8B"}
FEWSHOT_IDS = ["ID17", "ID350", "ID588", "ID113"]
print("F1 of predicting all 1:", round(f1_score(y_all, np.ones(len(y_all))), 3))

def evaluate(scores, masses, ids_exclude=()):
    \"\"\"Score one prompt (a Series indexed by id, higher = more insufficient) on the labelled questions.\"\"\"
    ids = [i for i in y_all.index if i not in ids_exclude]
    s, y = scores.loc[ids].values, y_all.loc[ids].values
    cv = cv_eval(LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED), s.reshape(-1, 1), y)
    return {"mass": masses.loc[ids].mean(), "AUC": roc_auc_score(y, s),
            "F1 zero-shot @0.5": f1_score(y, s >= 0.5, zero_division=0), "share pred 1": (s >= 0.5).mean(),
            "F1 CV tuned": cv["f1_tuned"], "F1 CV std": cv["f1_tuned_std"]}
""")

md("## Part 1 — wordings of the main question")

code("""
rows, V = [], {}
for tag, model_name in MODELS.items():
    f = Path(f"features/prompt_variants_{tag}.csv")
    if not f.exists():
        print("not available yet:", f); continue
    d = pd.read_csv(f).set_index("id"); V[tag] = d
    for v in [c for c in d.columns if not c.endswith("_mass")]:
        r = evaluate(d[v], d[f"{v}_mass"], FEWSHOT_IDS if v == "v9_fewshot" else ())
        rows.append({"model": model_name, "prompt": v, **r})
    # average of all zero-shot variants (no selection, so no optimistic bias)
    zs = [c for c in d.columns if c.startswith("v") and not c.endswith("_mass") and c != "v9_fewshot"]
    r = evaluate(d[zs].mean(1), d[[c + "_mass" for c in zs]].mean(1))
    rows.append({"model": model_name, "prompt": "mean of v1-v8", **r})
variants = pd.DataFrame(rows)
variants.round(3)
""")

code("""
# The full prompt texts, for reference
for tag in V:
    spec = json.load(open(f"features/prompt_variants_{tag}.json"))
    if tag == next(iter(V)):
        for k, v in spec["variants"].items():
            print(f"--- {k} ({v['kind']})\\nSYSTEM: {v['system']}\\nUSER:   {v['user']}\\n")
""")

code("""
# AUC per prompt and model (0.5 = no better than random ordering)
if len(variants):
    piv = variants.pivot(index="prompt", columns="model", values="AUC")
    piv = piv.loc[piv.mean(1).sort_values().index]
    colors = {"Qwen2.5-VL-7B": "#2a78d6", "Qwen3-8B": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(piv) + 1))
    h = 0.8 / len(piv.columns)
    for j, m in enumerate(piv.columns):
        ax.barh(np.arange(len(piv)) + (j - (len(piv.columns) - 1) / 2) * h, piv[m], height=h * 0.9,
                color=colors.get(m), label=m)
    ax.axvline(0.5, color="#52514e", lw=1, ls="--"); ax.text(0.505, -0.9, "random", color="#52514e", fontsize=8)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    ax.set_xlim(0.3, 0.9); ax.set_xlabel("AUC on 64 labelled questions")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right"); ax.set_title("Prompt wording vs ranking quality", loc="left")
    plt.tight_layout(); plt.show()
""")

md("## Part 2 — checklist prompts (`llm_judge.py`)")

code("""
rows, C = [], {}
for tag, model_name in MODELS.items():
    f = Path(f"features/llm_{tag}.csv")
    if not f.exists():
        print("not available yet:", f); continue
    d = pd.read_csv(f).set_index("id"); C[tag] = d
    for c in [c for c in d.columns if not c.endswith("_mass")]:
        # For prompts where "Yes" means GOOD information, flip so that higher = more insufficient
        positive = c in ("sufficient", "product_version", "clear_symptom", "error_text", "tried_steps")
        s = 1 - d[c] if positive else d[c]
        rows.append({"model": model_name, "prompt": c + (" (flipped)" if positive else ""), **evaluate(s, d[f"{c}_mass"])})
checklist = pd.DataFrame(rows)
checklist.round(3)
""")

md("""
## Part 3 — second prompt set (`prompts_v2.py`): plain-English questions

The first attempt (Parts 1-2) had awkward wordings ("Is this question sufficient?" — sufficient for what?,
capitals, "label 1" jargon, convoluted checklist items). The second set uses one template and plain questions:
W1-W9 = ways of asking the main question, C1-C12 = one specific piece of information each.
All scores again point the same way (higher = more likely insufficient).
""")

code("""
rows, P2 = [], {}
for tag, model_name in MODELS.items():
    f = Path(f"features/prompts_v2_{tag}.csv")
    if not f.exists():
        print("not available yet:", f); continue
    d = pd.read_csv(f).set_index("id"); P2[tag] = d
    spec = json.load(open(f"features/prompts_v2_{tag}.json"))["prompts"]
    for c in [c for c in d.columns if not c.endswith("_mass")]:
        rows.append({"model": model_name, "prompt": c, "question": spec[c]["question"],
                     **evaluate(d[c], d[f"{c}_mass"])})
    W = [c for c in d.columns if c.startswith("W") and not c.endswith("_mass")]
    Cs = [c for c in d.columns if c.startswith("C") and not c.endswith("_mass")]
    rows.append({"model": model_name, "prompt": "mean of W1-W9", "question": "(average, no selection)",
                 **evaluate(d[W].mean(1), d[[w + "_mass" for w in W]].mean(1))})
    rows.append({"model": model_name, "prompt": "mean of C1-C12", "question": "(average, no selection)",
                 **evaluate(d[Cs].mean(1), d[[c + "_mass" for c in Cs]].mean(1))})
v2 = pd.DataFrame(rows)
v2.round(3)
""")

code("""
# AUC per new prompt and model
if len(v2):
    piv = v2.pivot(index="prompt", columns="model", values="AUC")
    piv = piv.loc[piv.mean(1).sort_values().index]
    colors = {"Qwen2.5-VL-7B": "#2a78d6", "Qwen3-8B": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(piv) + 1))
    h = 0.8 / len(piv.columns)
    for j, m in enumerate(piv.columns):
        ax.barh(np.arange(len(piv)) + (j - (len(piv.columns) - 1) / 2) * h, piv[m], height=h * 0.9,
                color=colors.get(m), label=m)
    ax.axvline(0.5, color="#52514e", lw=1, ls="--")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    ax.set_xlim(0.3, 0.9); ax.set_xlabel("AUC on 64 labelled questions")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right"); ax.set_title("Second prompt set: ranking quality", loc="left")
    plt.tight_layout(); plt.show()
""")

code("""
# First attempt vs second set, per model: average quality of all prompts and of the best one
cmp = []
for name, df_ in [("1st attempt: wordings", variants), ("1st attempt: checklist", checklist), ("2nd set", v2)]:
    if not len(df_): continue
    single = df_[~df_.prompt.str.startswith("mean")]
    for m, g in single.groupby("model"):
        cmp.append({"set": name, "model": m, "n prompts": len(g), "mean AUC": g.AUC.mean(), "best AUC": g.AUC.max(),
                    "mean F1 CV": g["F1 CV tuned"].mean(), "best F1 CV": g["F1 CV tuned"].max(),
                    "n beating all-1 (0.51)": int((g["F1 CV tuned"] > 0.51).sum())})
pd.DataFrame(cmp).round(3)
""")

md("""
## Summary

* **worked** = AUC clearly above 0.5 *and* F1 CV above the all-1 reference (0.51);
* **did not work** = AUC near 0.5, or the model ignored the format (low mass), or it predicts
  (almost) everything as one class at 0.5 (see `share pred 1`).
""")

code("""
summary = pd.concat([variants.assign(group="1st wording"), checklist.assign(group="1st checklist"),
                     v2.assign(group="2nd set")], ignore_index=True)
summary["verdict"] = np.where((summary["AUC"] >= 0.6) & (summary["F1 CV tuned"] > 0.51), "worked",
                     np.where(summary["AUC"] >= 0.55, "weak", "did not work"))
summary.sort_values("AUC", ascending=False)[["group", "model", "prompt", "AUC", "F1 zero-shot @0.5",
                                             "share pred 1", "F1 CV tuned", "mass", "verdict"]].round(3)
""")

code("""
summary.to_csv("logs/prompt_results.csv", index=False)
print("saved logs/prompt_results.csv")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("prompts.ipynb", "w"), indent=1)
