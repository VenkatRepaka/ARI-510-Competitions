"""Extra features for the non-LLM experiments (nonllm.ipynb). No language model is asked anything here.

TEXT STATISTICS (`text_stats`)
-------------------------------
Simple counts that describe HOW a question is written rather than what it is about:
length, how many digits / capital letters / technical tokens it contains, whether it has a link,
code, file paths or quoted text, and how "specific" the vocabulary is. The idea: a question with
enough detail for troubleshooting usually contains concrete identifiers (version numbers, model names,
error codes, paths), while a vague one is short and made of common words.

TITLE / BODY EMBEDDINGS (`embed_parts`)
---------------------------------------
The same sentence encoders as before (mpnet, BERT), but applied to the title and the body separately.
A full-text embedding is dominated by the topic (Outlook, iPhone, Excel ...), which says little about
how complete the question is; embedding the parts separately lets a model compare them.
Vectors are cached in features/ so the notebook runs quickly.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from feats import embed, split_title

# Very common English words; a high share of these = generic wording, few concrete details.
COMMON = set("""the a an and or but if to of in on at for with from by is are was were be been being it this that
these those i you he she we they my your our their me him her them do does did have has had can could will would
should may might must not no yes so just also very really there here what when where why how which who all any
some more most other such only own same than too s t don now get got help please thanks thank hello hi anyone
know want need like one problem issue question work working""".split())

TECH = re.compile(r"\b(?:[A-Z]{2,}\d*|\w*\d\w*|\w+\.(?:exe|dll|xlsx?|docx?|pdf|csv|json|log|txt|zip|msi|dmg|app))\b")


def text_stats(texts):
    """Writing-style counts, one row per question (DataFrame)."""
    rows = []
    for t in texts:
        title, body = split_title(t)
        words = re.findall(r"[A-Za-z']+", body)
        lw = [w.lower() for w in words]
        n = max(len(words), 1)
        rows.append({
            "log_chars": np.log1p(len(body)),
            "log_title_chars": np.log1p(len(title)),
            "title_body_ratio": len(title) / max(len(body), 1),
            "share_digits": sum(c.isdigit() for c in body) / max(len(body), 1),
            "share_upper": sum(c.isupper() for c in body) / max(len(body), 1),
            "share_common_words": sum(w in COMMON for w in lw) / n,    # high = generic wording
            "share_long_words": sum(len(w) >= 8 for w in lw) / n,
            "type_token_ratio": len(set(lw)) / n,                      # vocabulary variety
            "n_tech_tokens": np.log1p(len(TECH.findall(body))),        # e.g. 0x80070005, KB5034441, iOS17, report.xlsx
            "has_url": int(bool(re.search(r"https?://|www\.", body))),
            "has_path": int(bool(re.search(r"[A-Za-z]:\\|/\w+/\w+|%\w+%", body))),
            "has_code": int(bool(re.search(r"[{};=]|\(\)|=\w+\(", body))),
            "n_newlines": np.log1p(body.count("\n")),
            "n_exclaim": body.count("!"),
            "starts_with_greeting": int(bool(re.match(r"\s*(hi|hello|hey|dear|good (morning|afternoon))", body.lower()))),
            "title_is_question": int(title.strip().endswith("?")),
        })
    return pd.DataFrame(rows)


def embed_parts(texts, model_name, tag, cache_dir="features"):
    """Title and body embeddings (cached as features/emb_<tag>_title.npy / _body.npy)."""
    out = {}
    for part, idx in (("title", 0), ("body", 1)):
        f = Path(cache_dir) / f"emb_{tag}_{part}.npy"
        if not f.exists():
            np.save(f, embed([split_title(t)[idx] or " " for t in texts], model_name))
        out[part] = np.load(f)
    return out["title"], out["body"]
