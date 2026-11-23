"""Zero-shot LLM checklist for every train + test question (no labels are shown to the model).

WHAT THIS SCRIPT DOES
---------------------
A local language model (LLM) reads each of the 114 tech-support questions (64 train + 50 test) and
answers a few Yes/No questions about it:
  * one overall question:  "Does this question give enough information to start troubleshooting?"
  * a checklist of things a support person usually needs (exact error message, product/version,
    clear symptom, what was already tried, ...).
Each answer is stored as a soft probability P(Yes) in features/llm_<tag>.csv. The notebook
models.ipynb then uses these numbers as features for a small classifier that is trained on the
64 labelled questions.

READING PROBABILITIES INSTEAD OF GENERATING TEXT
------------------------------------------------
Before an LLM writes its first answer token it assigns a probability to every token it knows.
The script runs the model ONCE per question and compares the probabilities of "Yes" and "No":
    score = P(Yes) / (P(Yes) + P(No))
e.g. P(Yes)=0.30, P(No)=0.60  ->  score 0.33. No sampling, so every run gives identical numbers.
`mass` = P(Yes)+P(No) is stored too: close to 1 means the model really answered Yes/No.

Usage: python llm_judge.py [model_id] [tag]
  default: mlx-community/Qwen2.5-VL-7B-Instruct-4bit -> features/llm_q25.csv (+ features/llm_prompts.json)
  e.g.     python llm_judge.py mlx-community/Qwen3-8B-4bit q3
Both models are Apache-2.0 and run locally with MLX (text only).
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
TAG = sys.argv[2] if len(sys.argv) > 2 else "q25"
OUT = Path("features"); OUT.mkdir(exist_ok=True)

# The system message sets the role. The definition of "sufficient" is copied from the competition's
# data description, so the model judges with the same rule the annotators were given.
SYSTEM = ("You are an experienced IT support engineer reviewing questions posted on a tech-support forum. "
          "A question is SUFFICIENT if it provides enough relevant details to understand the issue and begin "
          "troubleshooting; it is INSUFFICIENT if it lacks important details needed for effective troubleshooting. "
          "Answer only with Yes or No.")

# {q} is replaced by the full question text (title + body).
PROMPTS = {
    # the overall judgement (the main feature)
    "sufficient": "Support question:\n\"\"\"\n{q}\n\"\"\"\nDoes this question provide enough information to understand the issue and begin troubleshooting? Answer Yes or No.",
    "need_more": "Support question:\n\"\"\"\n{q}\n\"\"\"\nWould you have to ask the poster for more details before you could start helping? Answer Yes or No.",
    # checklist items: what a support engineer usually looks for
    "product_version": "Support question:\n\"\"\"\n{q}\n\"\"\"\nDoes the question name the specific product AND its version, device model or operating system? Answer Yes or No.",
    "clear_symptom": "Support question:\n\"\"\"\n{q}\n\"\"\"\nDoes the question clearly describe what goes wrong (or what the user wants to achieve) in concrete terms? Answer Yes or No.",
    "error_text": "Support question:\n\"\"\"\n{q}\n\"\"\"\nIf an error or failure is mentioned, is the exact error message or behaviour quoted? (Answer Yes if no error is involved.) Answer Yes or No.",
    "tried_steps": "Support question:\n\"\"\"\n{q}\n\"\"\"\nDoes the question say what the user has already tried or the steps that lead to the problem? Answer Yes or No.",
    "missing_attachment": "Support question:\n\"\"\"\n{q}\n\"\"\"\nDoes the question refer to a file, screenshot, data, formula or error that is NOT actually shown in the text? Answer Yes or No.",
    "vague": "Support question:\n\"\"\"\n{q}\n\"\"\"\nIs the question vague, very short, or ambiguous about what exactly the problem is? Answer Yes or No.",
}


def main():
    """Ask every prompt about every question and save the soft Yes-probabilities."""
    # ---- load model -> (tokenizer, forward function: token ids -> logits) ----
    if "VL" in MODEL:
        from mlx_vlm import load
        model, proc = load(MODEL)
        tok, lm = proc.tokenizer, model.language_model

        def forward(ids):
            # mlx-vlm keeps the position info (RoPE) of the first prompt and reuses it for later
            # text-only prompts; clearing it before each call keeps every question independent.
            lm._rope_deltas = None
            lm._position_ids = None
            out = lm(ids)
            return getattr(out, "logits", out)
    else:
        from mlx_lm import load
        model, tok = load(MODEL)
        forward = model
    # Qwen3 would otherwise write a <think> block first; this makes the first token the answer.
    template_kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}

    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    def p_yes(user_msg):
        """One forward pass; returns (P(Yes)/(P(Yes)+P(No)), P(Yes)+P(No)) for the first answer token."""
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kw)
        logits = forward(mx.array([tok.encode(s)]))[0, -1].astype(mx.float32)
        p = np.array(mx.softmax(logits))
        return float(p[yes_id] / (p[yes_id] + p[no_id])), float(p[yes_id] + p[no_id])

    # Train and test are scored together; the label column is never read.
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")], ignore_index=True)
    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        row = {"id": r.id}
        for name, tmpl in PROMPTS.items():
            row[name], row[f"{name}_mass"] = p_yes(tmpl.format(q=r.text.strip()))
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} questions, {time.time() - t0:.0f}s", flush=True)

    out_csv = OUT / f"llm_{TAG}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    json.dump({"model": MODEL, "system": SYSTEM, "prompts": PROMPTS,
               "readout": "next-token softmax, P(Yes)/(P(Yes)+P(No))"},
              open(OUT / f"llm_prompts_{TAG}.json", "w"), indent=2)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
