"""Pairwise LLM comparisons -> Bradley-Terry interest score per title (zero-shot, no labels shown).

Each of the 150 titles (train + test) is compared with K random other titles. Every pair is asked in BOTH orders
(A/B swapped) to cancel the model's position bias. Readout: P("A") / (P("A") + P("B")) from the next-token
softmax, so each comparison gives a soft win probability. A Bradley-Terry model (P(i beats j) = sigmoid(s_i - s_j),
small L2 penalty) turns the soft wins into one score per title.

Usage: python llm_pairwise.py <model_id> <tag> [K]
  writes features/pairwise_<tag>.csv (every comparison) and features/pairwise_scores_<tag>.csv (title scores)
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

MODEL, TAG = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SEED = 42
OUT = Path("features"); OUT.mkdir(exist_ok=True)

KIND = {"youtube": "YouTube video", "reddit": "Reddit post", "podcast": "podcast episode", "article": "news article"}
SYSTEM = "You are an average adult internet user judging content titles. Answer only with what is asked."
PROMPT = ("Which of these two titles would a typical reader find more interesting?\n"
          "A ({kind_a}): {title_a}\n"
          "B ({kind_b}): {title_b}\n"
          "Answer with a single letter: A or B.")


def load_model(name):
    """Load the LLM and return (forward, tokenizer); forward(ids) gives next-token logits.
    For mlx-vlm models the cached RoPE positions are reset before every call (see llm_rate.py).
    """
    if "VL" in name:
        from mlx_vlm import load
        model, proc = load(name)
        lm = model.language_model

        def forward(ids):
            """Return next-token logits; clears mlx-vlm's cached RoPE positions so each prompt is independent."""
            lm._rope_deltas = None      # mlx-vlm reuses the first prompt's RoPE positions otherwise
            lm._position_ids = None
            out = lm(ids)
            return getattr(out, "logits", out)
        return forward, proc.tokenizer
    from mlx_lm import load
    model, tok = load(name)
    return model, tok


def make_pairs(n, k, seed=SEED):
    """Choose which titles to compare: keep drawing random opponents until every title i appears
    in at least k distinct pairs. Pairs are stored as (smaller index, larger index) so (3, 7) and (7, 3)
    count once; the two display orders are handled later. Fixed seed -> same pairs every run.
    """
    rng = np.random.default_rng(seed)
    pairs = set()
    for i in range(n):
        while sum(1 for p in pairs if i in p) < k:
            j = int(rng.integers(n))
            if j != i:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def bradley_terry(n, i_idx, j_idx, p_i, l2=0.01):
    """Turn many pairwise results into one strength score per title (Bradley-Terry model).

    Model: P(title i beats title j) = sigmoid(s_i - s_j); a bigger s means 'more interesting'.
    Inputs:  i_idx, j_idx = the two titles of each comparison; p_i = soft probability that i won (0..1).
    Fit:     find the scores s that minimise cross-entropy between p_i and sigmoid(s_i - s_j),
             plus a small L2 penalty l2 * sum(s^2) that keeps scores finite and pulls rarely-compared
             titles toward 0. Solved with L-BFGS using the analytic gradient.
    Returns s centred at mean 0. Advantage over a plain win rate: beating a strong opponent counts more.
    """
    from scipy.optimize import minimize

    def f(s):
        """Loss and gradient for the optimiser.
        q = predicted P(i beats j); the gradient of cross-entropy w.r.t. s_i is (q - p_i) and w.r.t. s_j is (p_i - q);
        np.add.at sums these over all comparisons a title took part in.
        """
        d = s[i_idx] - s[j_idx]
        q = 1 / (1 + np.exp(-d))
        loss = -(p_i * np.log(q + 1e-12) + (1 - p_i) * np.log(1 - q + 1e-12)).sum() + l2 * (s ** 2).sum()
        g = np.zeros(n)
        np.add.at(g, i_idx, q - p_i)
        np.add.at(g, j_idx, p_i - q)
        return loss, g + 2 * l2 * s
    s = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B").x
    return s - s.mean()


def main():
    """Compare every pair in both orders with the LLM, then score titles.
    1. P(A) for 'title i shown as A' and for 'title j shown as A'.
    2. p_i = average of P(i first) and 1 - P(j first): cancels the model's habit of preferring A (position bias).
    3. Save every comparison, then per title: Bradley-Terry score and soft win rate (average of its p's).
    """
    forward, tok = load_model(MODEL)
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}

    def p_first(ta, da, tb, db):
        """Ask 'which title is more interesting, A or B?' and read the next-token probabilities.
        Returns (P(A) / (P(A) + P(B)), P(A) + P(B)); the second value checks the model really answers A or B.
        """
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(kind_a=KIND[da], title_a=ta, kind_b=KIND[db], title_b=tb)}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        logits = forward(mx.array([tok.encode(s)]))[0, -1].astype(mx.float32)
        p = np.array(mx.softmax(logits))
        return float(p[a_id] / (p[a_id] + p[b_id])), float(p[a_id] + p[b_id])

    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")], ignore_index=True)
    pairs = make_pairs(len(df), K)
    print(f"{len(pairs)} pairs x 2 orders = {2 * len(pairs)} calls", flush=True)
    rows, t0 = [], time.time()
    for c, (i, j) in enumerate(pairs):
        ri, rj = df.iloc[i], df.iloc[j]
        p_ij, m1 = p_first(ri.title, ri.domain, rj.title, rj.domain)      # i shown as A
        p_ji, m2 = p_first(rj.title, rj.domain, ri.title, ri.domain)      # j shown as A
        rows.append(dict(i=i, j=j, id_i=ri.title_id, id_j=rj.title_id, p_i_first=p_ij, p_j_first=p_ji,
                         p_i=(p_ij + (1 - p_ji)) / 2, mass=min(m1, m2)))
        if (c + 1) % 100 == 0:
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / f"pairwise_{TAG}.csv", index=False)

    s = bradley_terry(len(df), comp.i.values, comp.j.values, comp.p_i.values)
    wins = np.zeros(len(df)); cnt = np.zeros(len(df))
    np.add.at(wins, comp.i.values, comp.p_i.values); np.add.at(wins, comp.j.values, 1 - comp.p_i.values)
    np.add.at(cnt, comp.i.values, 1); np.add.at(cnt, comp.j.values, 1)
    pd.DataFrame({"title_id": df.title_id, f"bt_{TAG}": s, f"winrate_{TAG}": wins / cnt, f"n_comp_{TAG}": cnt}) \
        .to_csv(OUT / f"pairwise_scores_{TAG}.csv", index=False)
    pos_bias = float(np.mean(comp.p_i_first + comp.p_j_first) / 2)
    json.dump({"model": MODEL, "system": SYSTEM, "prompt": PROMPT, "K": K, "pairs": len(pairs), "seed": SEED,
               "position_bias_mean_P(A)": pos_bias, "min_AB_mass": float(comp.mass.min())},
              open(OUT / f"pairwise_prompts_{TAG}.json", "w"), indent=2)
    print(f"wrote pairwise scores | mean P(A first) {pos_bias:.3f} | min A/B mass {comp.mass.min():.3f}")


if __name__ == "__main__":
    main()
