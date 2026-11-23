"""Prompt-wording experiment: the SAME question ("is this support question sufficient?") asked in different ways.

WHY
---
An LLM's answer can change a lot with small changes in wording. To find out which wording works for this
task, each variant below is asked about all 114 questions (64 train + 50 test) with the same local model
and the same read-out. prompts.ipynb then compares the variants on the 64 labelled questions.
Labels are never shown to the model (except in the few-shot variant, see below).

READ-OUT
--------
For Yes/No variants: one forward pass, then P(Yes) / (P(Yes) + P(No)) of the first answer token.
For the 1-5 scale variant: P("1").."P(5)", rescaled to sum 1, expected value.
Every variant is converted to the same direction, a score in [0, 1] where HIGHER = MORE LIKELY INSUFFICIENT
(the competition's class 1), so they can be compared directly:
    positive wording ("is it sufficient?")      -> score = 1 - P(Yes)
    negative wording ("is it missing details?") -> score = P(Yes)
    1-5 completeness scale                       -> score = (5 - expected) / 4
`mass` = how much probability went to the allowed answers (close to 1 = the model followed the format).

THE VARIANTS (what changes from one to the next)
-----------------------------------------------
  v1_bare        minimal question, no role, no definition
  v2_definition  adds the competition's own definition of sufficient / insufficient
  v3_negative    same definition, but asks the opposite question ("does it LACK details?")
  v4_engineer    role play: a support engineer deciding whether to ask follow-up questions
  v5_checklist   lists the details a support engineer needs, then asks for an overall Yes/No
  v6_strict      tells the model that most forum questions miss something and to be strict
  v7_annotator   frames the model as one of five human annotators labelling a dataset
  v8_scale       1-5 completeness rating instead of Yes/No
  v9_fewshot     v2 plus 4 labelled training examples (2 per class) shown before the question.
                 Those 4 questions (FEWSHOT_IDS) must be left out when this variant is evaluated.

Usage: python prompt_variants.py [model_id] [tag]
  default: mlx-community/Qwen2.5-VL-7B-Instruct-4bit, tag q25 -> features/prompt_variants_q25.csv
           and features/prompt_variants_q25.json (exact prompt texts, for the write-up)
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

# The competition's definition (from the Data page), reused by several variants.
DEFINITION = ("A question is SUFFICIENT if it provides enough relevant details to understand the issue and begin "
              "troubleshooting. It is INSUFFICIENT if it lacks important details needed for effective troubleshooting.")

# Training questions shown as examples in v9 (2 sufficient, 2 insufficient; all short).
FEWSHOT_IDS = {"ID17": "Yes", "ID350": "Yes", "ID588": "No", "ID113": "No"}

# Each variant: system message, user message ({q} = question text), answer type, and direction.
#   kind "yesno_pos": Yes = sufficient     -> score = 1 - P(Yes)
#   kind "yesno_neg": Yes = insufficient   -> score = P(Yes)
#   kind "scale5"   : 5 = complete         -> score = (5 - E) / 4
Q = 'Support question:\n"""\n{q}\n"""\n'
VARIANTS = {
    "v1_bare": dict(kind="yesno_pos", system="Answer only Yes or No.",
                    user=Q + "Is this question sufficient? Answer Yes or No."),
    "v2_definition": dict(kind="yesno_pos", system="You review tech-support forum questions. " + DEFINITION + " Answer only Yes or No.",
                          user=Q + "Is this question SUFFICIENT? Answer Yes or No."),
    "v3_negative": dict(kind="yesno_neg", system="You review tech-support forum questions. " + DEFINITION + " Answer only Yes or No.",
                        user=Q + "Does this question LACK important details needed for effective troubleshooting? Answer Yes or No."),
    "v4_engineer": dict(kind="yesno_neg", system="You are an experienced IT support engineer answering questions on a support forum. Answer only Yes or No.",
                        user=Q + "Before you could start troubleshooting, would you need to ask the poster follow-up questions for missing information? Answer Yes or No."),
    "v5_checklist": dict(kind="yesno_pos", system="You review tech-support forum questions. " + DEFINITION + " Answer only Yes or No.",
                         user=Q + "A support engineer usually needs: (1) the product and its version / device / operating system, "
                                  "(2) a concrete description of what goes wrong or what the user wants, (3) the exact error message if there is one, "
                                  "(4) what has already been tried or the steps that cause the problem, (5) any file or data the question refers to.\n"
                                  "Considering what this specific issue needs, does the question give enough of this information to begin troubleshooting? Answer Yes or No."),
    "v6_strict": dict(kind="yesno_pos", system="You review tech-support forum questions. " + DEFINITION +
                      " Be strict: many forum posts leave out something important (versions, error text, steps, context). Answer only Yes or No.",
                      user=Q + "Is this question SUFFICIENT? Answer Yes or No."),
    "v7_annotator": dict(kind="yesno_neg", system="You are one of five human annotators labelling a dataset of technical support questions. "
                         "Label 0 = sufficient information, label 1 = insufficient information. " + DEFINITION + " Answer only Yes or No.",
                         user=Q + "Would you give this question label 1 (insufficient information)? Answer Yes or No."),
    "v8_scale": dict(kind="scale5", system="You review tech-support forum questions. Answer with a single digit.",
                     user=Q + "How complete is the information in this question for starting to troubleshoot the issue?\n"
                              "1 = almost nothing to work with, 2 = major details missing, 3 = some important details missing, "
                              "4 = mostly complete, 5 = everything needed is there.\nAnswer with a single digit from 1 to 5."),
}


def fewshot_variant(train):
    """v9: v2 plus four labelled training examples written as earlier chat turns."""
    v = dict(VARIANTS["v2_definition"])
    shots = []
    for qid, ans in FEWSHOT_IDS.items():
        text = train.loc[train.id == qid, "text"].iloc[0].strip()
        shots += [{"role": "user", "content": v["user"].format(q=text)}, {"role": "assistant", "content": ans}]
    v["shots"] = shots
    return v


def main():
    # ---- load the model (same as llm_judge.py) ----
    if "VL" in MODEL:
        from mlx_vlm import load
        model, proc = load(MODEL)
        tok, lm = proc.tokenizer, model.language_model

        def forward(ids):
            lm._rope_deltas = None          # mlx-vlm bug work-around: forget the previous prompt's positions
            lm._position_ids = None
            out = lm(ids)
            return getattr(out, "logits", out)
    else:
        from mlx_lm import load
        model, tok = load(MODEL)
        forward = model
    template_kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}

    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]

    def probs(messages):
        """Probabilities of every token as the first token of the answer (one forward pass, no sampling)."""
        s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kw)
        logits = forward(mx.array([tok.encode(s)]))[0, -1].astype(mx.float32)
        return np.array(mx.softmax(logits))

    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    df = pd.concat([train, test], ignore_index=True)
    variants = dict(VARIANTS, v9_fewshot=fewshot_variant(train))

    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        row = {"id": r.id}
        for name, v in variants.items():
            msgs = [{"role": "system", "content": v["system"]}] + v.get("shots", []) + \
                   [{"role": "user", "content": v["user"].format(q=r.text.strip())}]
            p = probs(msgs)
            if v["kind"] == "scale5":
                d = p[digit_ids]
                row[f"{name}_mass"] = float(d.sum())
                e = float((d / d.sum() * np.arange(1, 6)).sum())     # expected rating 1..5
                row[name] = (5 - e) / 4                               # -> 0 (complete) .. 1 (nothing there)
            else:
                py = float(p[yes_id] / (p[yes_id] + p[no_id]))
                row[f"{name}_mass"] = float(p[yes_id] + p[no_id])
                row[name] = 1 - py if v["kind"] == "yesno_pos" else py
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} questions, {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"prompt_variants_{TAG}.csv", index=False)
    json.dump({"model": MODEL, "readout": "first-token softmax; score = P(insufficient) direction",
               "fewshot_ids": FEWSHOT_IDS,
               "variants": {k: {kk: vv for kk, vv in v.items() if kk != "shots"} for k, v in variants.items()}},
              open(OUT / f"prompt_variants_{TAG}.json", "w"), indent=2)
    print("wrote", OUT / f"prompt_variants_{TAG}.csv")


if __name__ == "__main__":
    main()
