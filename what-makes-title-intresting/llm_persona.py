"""Persona ensemble: the same local LLM rates every title as 6 different readers (zero-shot, no labels shown).

The test labels are the mean of 6 human annotators; averaging 6 simulated readers mimics that and reduces the
prompt-specific noise of a single rating. Same readout as llm_rate.py (softmax over digits 1-6 -> expected value).

Usage: python llm_persona.py [model_id] [out_csv]
  default model mlx-community/Qwen2.5-VL-7B-Instruct-4bit (Apache-2.0) -> features/llm_persona.csv
Prompts are saved to features/llm_persona_prompts.json.
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
OUT_CSV = Path(sys.argv[2] if len(sys.argv) > 2 else "features/llm_persona.csv")
OUT_CSV.parent.mkdir(exist_ok=True)

KIND = {"youtube": "YouTube video", "reddit": "Reddit post", "podcast": "podcast episode", "article": "news article"}

PERSONAS = {
    "student": "You are a 20-year-old university student who spends a lot of time on YouTube, Reddit and gaming.",
    "parent": "You are a 40-year-old working parent with little free time who reads news and listens to podcasts on the commute.",
    "retiree": "You are a 68-year-old retiree who reads newspapers daily and enjoys history, health and local news.",
    "professional": "You are a 32-year-old office professional interested in technology, science and self-improvement.",
    "casual": "You are an average adult who browses social media casually and is easily bored by dull or confusing titles.",
    "critic": "You are a skeptical media-literate reader who dislikes clickbait but enjoys genuinely surprising or useful stories.",
}

PROMPT = ("Here is the title of a {kind}.\nTitle: {title}\n"
          "How interesting is this title to you personally? "
          "Scale: 1 = not interesting at all, 2 = slightly, 3 = somewhat, 4 = fairly, 5 = very, 6 = extremely interesting.\n"
          "Answer with a single digit from 1 to 6.")


def load_model(name):
    """Load the LLM and return (forward, tokenizer).

    forward(ids) -> logits for the next token. For mlx-vlm models the cached RoPE positions are reset
    before every call (same bug fix as llm_rate.py).
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
    return (lambda ids: model(ids)), tok


def main():
    """Rate every title once per persona (6 simulated readers) and save one column per persona.
    Idea: the test labels average 6 human raters, so averaging 6 simulated readers might mimic that.
    (It did not improve the dev score; kept for documentation.)
    """
    forward, tok = load_model(MODEL)
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 7)]

    def rating(system, user):
        """One rating from one persona: returns (mass, expected rating).
        mass = total probability on the digits 1-6 (sanity check that the model answers with a digit);
        expected rating = sum(k * P(k)) after renormalising P(1..6).
        """
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        logits = forward(mx.array([tok.encode(s)]))[0, -1].astype(mx.float32)
        p = np.array(mx.softmax(logits))[digit_ids]
        return float(p.sum()), float((p / p.sum() * np.arange(1, 7)).sum())

    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")], ignore_index=True)
    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        row = {"title_id": r.title_id}
        for name, persona in PERSONAS.items():
            mass, e = rating(persona, PROMPT.format(kind=KIND[r.domain], title=r.title))
            row[f"persona_{name}"] = e
            row[f"mass_{name}"] = mass
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(df)} titles, {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out["persona_mean"] = out[[f"persona_{k}" for k in PERSONAS]].mean(1)
    out.to_csv(OUT_CSV, index=False)
    json.dump({"model": MODEL, "personas": PERSONAS, "prompt": PROMPT, "kind_names": KIND,
               "readout": "next-token softmax over digits 1-6 -> expected value; persona_mean = average of 6"},
              open(OUT_CSV.with_name(OUT_CSV.stem + "_prompts.json"), "w"), indent=2)
    print("wrote", OUT_CSV, "| min digit mass", round(out.filter(like="mass_").values.min(), 3))


if __name__ == "__main__":
    main()
