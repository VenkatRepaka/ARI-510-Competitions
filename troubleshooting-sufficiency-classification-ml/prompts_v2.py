"""Second prompt set: plain-English questions, one template, one idea per prompt.

WHY A SECOND SET
----------------
The first attempt (llm_judge.py + prompt_variants.py) had awkward wordings: "Is this question sufficient?"
(sufficient for what?), capitals ("SUFFICIENT"), dataset jargon ("label 1"), and convoluted checklist
questions. This set asks every question the way a person would, with the same template each time:

    system:  You are helping review questions posted on a tech-support forum. Answer only Yes or No.
    user:    Support question:
             \"\"\"<title + question text>\"\"\"
             <one Yes/No question>

Two prompt groups:
  * W1-W9  different ways of asking the MAIN question (is there enough information?)
  * C1-C12 checklist items: one specific piece of information each

READ-OUT (same as before)
-------------------------
One forward pass per prompt, no text generated. From the probabilities of the first answer token:
    P(Yes) / (P(Yes) + P(No))
Then every prompt is turned into the same direction, a score where HIGHER = MORE LIKELY INSUFFICIENT:
    "yes_is_good" prompts (Yes = information is there)   -> score = 1 - P(Yes)
    "yes_is_bad"  prompts (Yes = something is missing)   -> score = P(Yes)
Column <name> holds that score; <name>_mass = P(Yes)+P(No) (close to 1 = the model answered Yes/No).

Usage: python prompts_v2.py [model_id] [tag]
  default Qwen2.5-VL-7B-Instruct-4bit, tag q25 -> features/prompts_v2_q25.csv + features/prompts_v2_q25.json
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

SYSTEM = "You are helping review questions posted on a tech-support forum. Answer only Yes or No."
SYSTEM_ROLE = "You are an experienced IT support technician. Answer only Yes or No."
DEFINITION = ("A question has sufficient information if it gives enough relevant details to understand the issue "
              "and begin troubleshooting. It has insufficient information if important details needed for "
              "troubleshooting are missing.")
DIRECT = "Does this question give enough information to start troubleshooting the problem?"

# name: (direction, question, system message)
PROMPTS = {
    # ---- W: ways of asking the main question ----
    "W1_direct":     ("yes_is_good", DIRECT, SYSTEM),
    "W2_missing":    ("yes_is_bad", "Is important information missing from this question?", SYSTEM),
    "W3_followup":   ("yes_is_bad", "If you were answering this, would you need to ask the person for more details first?", SYSTEM),
    "W4_reproduce":  ("yes_is_good", "Could you try to reproduce or diagnose this problem using only what is written here?", SYSTEM),
    "W5_role":       ("yes_is_good", DIRECT, SYSTEM_ROLE),
    "W6_definition": ("yes_is_good", DIRECT, SYSTEM + " " + DEFINITION),
    "W7_expert":     ("yes_is_good", "Could an expert reply with a useful answer straight away, without asking anything back?", SYSTEM),
    "W8_next_step":  ("yes_is_good", "Is it clear what the first troubleshooting step should be?", SYSTEM),
    "W9_cause":      ("yes_is_good", "Can you tell what is most likely causing this problem from the information given?", SYSTEM),
    # ---- C: checklist, one piece of information each ----
    "C1_product":    ("yes_is_good", "Does the question say which product, device or software version is involved?", SYSTEM),
    "C2_symptom":    ("yes_is_good", "Does the question describe exactly what happens, or what the user is trying to do?", SYSTEM),
    "C3_error":      ("yes_is_good", "Does the question include the actual error message (if there is one)?", SYSTEM),
    "C4_tried":      ("yes_is_good", "Does the question say what the user has already tried?", SYSTEM),
    "C5_not_included": ("yes_is_bad", "Does the question mention a file, screenshot or data that is not included in the text?", SYSTEM),
    "C6_vague":      ("yes_is_bad", "Is the question too short or too vague to act on?", SYSTEM),
    # how-to questions are often short but still answerable, so Yes here points to "sufficient"
    "C7_howto":      ("yes_is_good", "Is this a how-to or general information question rather than a report of something going wrong?", SYSTEM),
    "C8_expected":   ("yes_is_good", "Does the question say what the user expected to happen and what actually happened?", SYSTEM),
    "C9_when":       ("yes_is_good", "Does the question say when the problem started or what changed before it (an update, a new device, a setting)?", SYSTEM),
    "C10_scope":     ("yes_is_good", "Does the question say whether the problem happens everywhere or only on one device, app or account?", SYSTEM),
    "C11_one_problem": ("yes_is_good", "Is the question about one clear problem, rather than several unrelated complaints?", SYSTEM),
    "C12_repeat":    ("yes_is_good", "Does the question say whether the problem happens every time or only sometimes?", SYSTEM),
}


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

    def p_yes(system, user):
        """One forward pass -> (P(Yes)/(P(Yes)+P(No)), P(Yes)+P(No)) for the first answer token."""
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kw)
        p = np.array(mx.softmax(forward(mx.array([tok.encode(s)]))[0, -1].astype(mx.float32)))
        return float(p[yes_id] / (p[yes_id] + p[no_id])), float(p[yes_id] + p[no_id])

    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")], ignore_index=True)
    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        row = {"id": r.id}
        for name, (direction, question, system) in PROMPTS.items():
            py, mass = p_yes(system, f'Support question:\n"""\n{r.text.strip()}\n"""\n{question}')
            row[name] = 1 - py if direction == "yes_is_good" else py
            row[f"{name}_mass"] = mass
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} questions, {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"prompts_v2_{TAG}.csv", index=False)
    json.dump({"model": MODEL, "template": 'Support question:\\n"""\\n{text}\\n"""\\n{question}',
               "readout": "first-token softmax; score = 1-P(Yes) for yes_is_good, P(Yes) for yes_is_bad (higher = insufficient)",
               "prompts": {k: {"direction": d, "question": q, "system": s} for k, (d, q, s) in PROMPTS.items()}},
              open(OUT / f"prompts_v2_{TAG}.json", "w"), indent=2)
    print("wrote", OUT / f"prompts_v2_{TAG}.csv")


if __name__ == "__main__":
    main()
