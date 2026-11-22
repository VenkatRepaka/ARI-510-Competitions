"""Zero-shot Claude ratings for every train + test title (no labels are shown to the model).

Model: claude-opus-5-5 via the official Anthropic SDK; the API key is read from ANTHROPIC_API_KEY (never stored here).
The Claude API does not return token probabilities, so instead of reading P("1".."6") from logits (llm_rate.py),
Claude returns a probability for each score 1-6 as structured JSON (output_config.format = JSON schema);
we normalise it and take the expected rating `claude_exp`. Same "typical reader" wording as llm_rate.py.
Effort "low" (thinking cannot be disabled on this model). Server-side refusal fallback enabled ("default").
Results are appended to the CSV as they arrive, so a rerun only scores missing titles (no double billing).

Usage: ANTHROPIC_API_KEY=... python claude_rate.py [limit]   -> features/claude_scores.csv
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

MODEL = "claude-opus-5-5"
OUT = Path("features/claude_scores.csv")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

KIND = {"youtube": "YouTube video", "reddit": "Reddit post", "podcast": "podcast episode", "article": "news article"}
SYSTEM = ("You are simulating a panel of average adult internet users judging how interesting content titles are. "
          "Judge only from the title and its source type.")
PROMPT = ("How interesting would a typical reader find this {kind} title?\n"
          "Scale: 1 = not interesting at all, 2 = slightly, 3 = somewhat, 4 = fairly, 5 = very, 6 = extremely interesting.\n"
          "Title: {title}\n"
          "Different readers disagree, so give the probability that a randomly chosen typical reader would pick each "
          "score (p1..p6, each between 0 and 1, summing to 1).")
SCHEMA = {
    "type": "object",
    "properties": {f"p{k}": {"type": "number"} for k in range(1, 7)},
    "required": [f"p{k}" for k in range(1, 7)],
    "additionalProperties": False,
}


def rate(client, title, domain):
    """Ask Claude for a probability distribution over the scores 1-6 for one title.

    The API gives no token probabilities, so instead the output is forced into a JSON schema
    {p1..p6} (structured output). Negative values are clipped to 0 and the six numbers are
    renormalised to sum to 1. Returns (probabilities, raw response), or (None, response) if refused.
    effort="low" keeps cost and time down; fallbacks="default" lets the server retry a refused
    request with a fallback model.
    """
    r = client.beta.messages.create(
        model=MODEL,
        max_tokens=2048,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(kind=KIND[domain], title=title)}],
    )
    if r.stop_reason == "refusal":
        return None, r
    p = json.loads(next(b.text for b in r.content if b.type == "text"))
    p = np.clip(np.array([p[f"p{k}"] for k in range(1, 7)], dtype=float), 0, None)
    p = p / p.sum()
    return p, r


def main():
    """Score every train + test title that is not already in the output CSV.

    Runs 4 API calls in parallel (thread pool). Each result is appended to the CSV immediately, so a
    crash or rerun never pays twice for the same title. Stores the expected rating
    claude_exp = sum(k * p_k), the six probabilities, and which model served the answer.
    At the end prints token usage / estimated cost and saves the prompt to features/claude_prompts.json.
    """
    client = anthropic.Anthropic(max_retries=4)
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")], ignore_index=True)
    done = set(pd.read_csv(OUT).title_id) if OUT.exists() else set()
    todo = df[~df.title_id.isin(done)]
    if LIMIT:
        todo = todo.head(LIMIT)
    print(f"{len(done)} already scored, scoring {len(todo)}", flush=True)
    tok_in = tok_out = 0
    rows = []
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(rate, client, r.title, r.domain): r.title_id for r in todo.itertuples()}
        for f in as_completed(futs):
            tid = futs[f]
            try:
                p, resp = f.result()
            except anthropic.APIStatusError as e:
                print("API error", tid, e.status_code, str(e)[:150], flush=True); continue
            tok_in += resp.usage.input_tokens; tok_out += resp.usage.output_tokens
            if p is None:
                print("refused", tid, flush=True); continue
            row = {"title_id": tid, "claude_exp": float((p * np.arange(1, 7)).sum()), "served_by": resp.model}
            row.update({f"claude_p{k + 1}": float(p[k]) for k in range(6)})
            rows.append(row)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"scored {len(rows)} | tokens in {tok_in} out {tok_out} | est. cost ${tok_in * 4e-6 + tok_out * 20e-6:.2f}")
    json.dump({"model": MODEL, "system": SYSTEM, "prompt": PROMPT, "schema": SCHEMA, "effort": "low",
               "readout": "structured JSON probabilities p1..p6, normalised -> expected value"},
              open("features/claude_prompts.json", "w"), indent=2)


if __name__ == "__main__":
    main()
