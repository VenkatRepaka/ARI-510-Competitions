"""Zero-shot LLM ratings for every train + test title (no labels are shown to the model).

WHAT THIS SCRIPT DOES
---------------------
A local language model (LLM) answers 7 questions about each of the 150 titles (120 train + 30 test):
  * 2 rating questions  ("How interesting is this title?", "Would you click it?")  -> answer 1..6
  * 5 trait questions   ("Does the title use emotional language?", ...)             -> answer Yes / No
The answers are saved as numeric features in features/llm_scores.csv. A later notebook (llm.ipynb)
uses these features in a ridge regression to predict the real interest score.

THE KEY IDEA: READ PROBABILITIES, DO NOT GENERATE TEXT
------------------------------------------------------
Before an LLM writes a token, it gives a probability to every token it knows. The script asks the
question, runs the model ONCE, and reads these probabilities for the first token of the answer.
No text is generated. This has three advantages:
  * more information: "P(3)=0.2, P(4)=0.6, P(5)=0.2" says more than the single answer "4"
  * deterministic: no random sampling, so every run gives the same numbers
  * fast: one forward pass per question

WORKED EXAMPLE (numbers made up for illustration)
-------------------------------------------------
Title "I quit my job to travel the world" (YouTube), interest question:
  probabilities of the tokens "1".."6":  0.01  0.04  0.20  0.50  0.20  0.03   (sum 0.98)
  interest_mass = 0.98   -> 98 % of the probability went to a digit, so the model followed the instruction
  rescaled to sum 1, then interest_exp = 1*p1 + 2*p2 + ... + 6*p6 ≈ 3.9   -> the main feature
For a Yes/No question: trait_emotional = P(Yes) / (P(Yes) + P(No)), e.g. 0.7 = "70 % yes".

OUTPUT COLUMNS (one row per title)
----------------------------------
  title_id
  interest_exp, click_exp       expected rating (weighted average of 1..6), range 1..6
  interest_p1..p6, click_p1..p6 the six rescaled probabilities (they sum to 1)
  interest_mass, click_mass     probability that landed on "1".."6" before rescaling, range 0..1;
                                close to 1 = the model really answered with a digit (a quality check)
  trait_<name>                  soft "yes" probability, range 0..1

MODEL
-----
Default: mlx-community/Qwen2.5-VL-7B-Instruct-4bit (Apache-2.0), run locally on the Mac with MLX, text only.

Usage: python llm_rate.py [model_id] [out_csv]
  default: Qwen2.5-VL-7B -> features/llm_scores.csv (+ features/llm_prompts.json)
  e.g.     python llm_rate.py mlx-community/Qwen3-8B-4bit features/llm_scores_qwen3_8b.csv
  Text-only models load with mlx-lm; Qwen3 runs with thinking disabled (enable_thinking=False).
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Settings from the command line (both optional)
# ---------------------------------------------------------------------------------------------
MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
OUT_CSV = Path(sys.argv[2] if len(sys.argv) > 2 else "features/llm_scores.csv")
OUT = OUT_CSV.parent; OUT.mkdir(exist_ok=True)   # create the output folder if it does not exist

# ---------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------
# The data stores the domain as a short code; the model reads a plain-English name instead.
KIND = {"youtube": "YouTube video", "reddit": "Reddit post", "podcast": "podcast episode", "article": "news article"}

# The system message sets the role: the model should judge like an ordinary reader.
SYSTEM = "You are an average adult internet user judging content titles. Answer only with what is asked."

# Rating questions: the answer is ONE digit 1..6 (the same scale as the competition labels).
# {kind} and {title} are filled in for every title with str.format().
RATING_PROMPTS = {
    "interest": ("How interesting would a typical reader find this {kind} title?\n"
                 "Scale: 1 = not interesting at all, 2 = slightly, 3 = somewhat, 4 = fairly, 5 = very, 6 = extremely interesting.\n"
                 "Title: {title}\nAnswer with a single digit from 1 to 6."),
    "click": ("You are scrolling past this {kind}. How likely are you to click or open it, based only on its title?\n"
              "Scale: 1 = would never click ... 6 = would definitely click.\n"
              "Title: {title}\nAnswer with a single digit from 1 to 6."),
}

# Trait questions: the answer is Yes or No. Each one checks a typical clickbait property.
TRAIT_PROMPTS = {
    "curiosity_gap": "Does this {kind} title deliberately hold back key information to make the reader curious (a curiosity gap)?\nTitle: {title}\nAnswer Yes or No.",
    "emotional": "Does this {kind} title use emotional or emotionally charged language?\nTitle: {title}\nAnswer Yes or No.",
    "exaggeration": "Does this {kind} title exaggerate or use hype (e.g. ALL CAPS, superlatives, sensational claims)?\nTitle: {title}\nAnswer Yes or No.",
    "direct_address": "Does this {kind} title directly address the reader (e.g. 'you', 'your', a question to the reader)?\nTitle: {title}\nAnswer Yes or No.",
    "missing_details": "Is this {kind} title missing details a reader would need to understand what it is about (vague or unclear without context)?\nTitle: {title}\nAnswer Yes or No.",
}


def main():
    """Rate every train + test title with a local LLM and save the scores (zero-shot: labels are never shown).

    Steps:
      1. Load the model and its tokenizer (the tokenizer turns text into token ids and back).
      2. Look up the token ids of "1".."6", "Yes" and "No".
      3. For each title and each of the 7 questions, run ONE forward pass and read the next-token probabilities:
           rating question -> P("1")..P("6"), rescaled to sum 1 -> expected rating   (column <name>_exp)
           trait question  -> P(Yes) / (P(Yes) + P(No))                            (column trait_<name>)
      4. Write one CSV row per title, plus a JSON file with the exact prompts (for the write-up).
    """
    # ---- Step 1: load the model --------------------------------------------------------------
    # "VL" models (vision-language, e.g. Qwen2.5-VL) load with mlx_vlm; only their text part is used.
    # Text-only models (e.g. Qwen3) load with mlx_lm.
    # Both branches end with the same two things:
    #   tok      = the tokenizer
    #   forward  = a function: token ids -> logits (one raw score per vocabulary token, per position)
    if "VL" in MODEL:
        from mlx_vlm import load
        model, proc = load(MODEL)
        tok, lm = proc.tokenizer, model.language_model

        def forward(ids):
            """Run the language model on token ids and return the logits (scores for every possible next token).

            The two resets work around an mlx-vlm bug: the library stores the RoPE position info
            (how the model knows the order of the tokens) of the FIRST prompt and reuses it for all
            later prompts. It only clears this when an image is passed, and here no image is passed.
            Without the reset, every title after the first gets wrong digit probabilities.
            Every question is independent, so the stored info is cleared before each call.
            """
            lm._rope_deltas = None
            lm._position_ids = None
            out = lm(ids)
            return getattr(out, "logits", out)   # some versions return an object, others the plain array
    else:
        from mlx_lm import load
        model, tok = load(MODEL)
        forward = model                           # an mlx_lm model can be called directly: ids -> logits

    # Qwen3 normally "thinks" first (it writes a <think> block before the answer). Then the first
    # token would not be the digit. enable_thinking=False makes the answer start immediately.
    template_kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}

    # ---- Step 2: token ids of the possible answers --------------------------------------------
    # The model works with numbers, not text. Each answer ("1", ..., "6", "Yes", "No") is one token;
    # its id is the position of that token in the probability vector.
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 7)]
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    def next_token_probs(user_msg):
        """Return the probability of every vocabulary token as the FIRST token of the model's answer.

        1. Put the system message and the question into the model's chat format
           (add_generation_prompt=True adds the "assistant:" part, so the next token is the answer).
        2. Encode the text into token ids and run the model once.
        3. [0, -1] keeps the scores at the LAST position, i.e. the prediction for the next token.
        4. softmax turns the raw scores into probabilities that sum to 1.
        No text is generated or sampled, so the result is always the same.
        """
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kw)
        logits = forward(mx.array([tok.encode(s)]))[0, -1].astype(mx.float32)
        return np.array(mx.softmax(logits))

    # ---- Step 3: ask all questions for all titles ---------------------------------------------
    # Train and test titles are rated together; no label column is used anywhere.
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")], ignore_index=True)
    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        fmt = dict(kind=KIND[r.domain], title=r.title)    # values for the {kind} and {title} blanks
        row = {"title_id": r.title_id}

        # Rating questions -> expected rating, the 6 probabilities, and the mass check
        for name, tmpl in RATING_PROMPTS.items():
            p = next_token_probs(tmpl.format(**fmt))[digit_ids]   # keep only P("1")..P("6")
            row[f"{name}_mass"] = float(p.sum())                  # how much probability landed on 1..6 (quality check)
            p = p / p.sum()                                       # rescale so the six values sum to 1
            row[f"{name}_exp"] = float((p * np.arange(1, 7)).sum())   # expected rating = 1*p1 + 2*p2 + ... + 6*p6
            for d in range(6):
                row[f"{name}_p{d + 1}"] = float(p[d])

        # Trait questions -> soft "yes" probability (only Yes and No are compared)
        for name, tmpl in TRAIT_PROMPTS.items():
            p = next_token_probs(tmpl.format(**fmt))
            row[f"trait_{name}"] = float(p[yes_id] / (p[yes_id] + p[no_id]))

        rows.append(row)
        if (i + 1) % 25 == 0:                                     # progress message every 25 titles
            print(f"{i + 1}/{len(df)} titles, {time.time() - t0:.0f}s", flush=True)

    # ---- Step 4: save ---------------------------------------------------------------------------
    # The CSV holds the features. The JSON (e.g. llm_scores.csv -> llm_prompts.json) records the model
    # and the exact prompt texts, so the write-up can quote them and the run can be repeated.
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    json.dump({"model": MODEL, "system": SYSTEM, "rating_prompts": RATING_PROMPTS, "trait_prompts": TRAIT_PROMPTS,
               "kind_names": KIND, "readout": "next-token softmax over digits 1-6 (expected value) / Yes vs No"},
              open(OUT_CSV.with_name(OUT_CSV.stem.replace("scores", "prompts") + ".json"), "w"), indent=2)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
