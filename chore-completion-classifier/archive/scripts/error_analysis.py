"""Which training photos does CV get wrong, and how often?

Uses only the original photos (web images ignored). For several reference models, runs the same
10x repeated 5-fold CV as train.py and records, per photo, how often it was misclassified and as what.
Writes logs/error_analysis.csv and logs/error_sheet_<chore>.jpg (wrongly classified photos with labels).

Usage: python error_analysis.py
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import RepeatedStratifiedKFold

import train as T
from extract_features import CLASSES, IMG_DIR, LOG_DIR, image_paths

MODELS = [  # (backbones, mode, classifier); several backbones = averaged probabilities
    (["dinov2_l", "eva02_l", "siglip_so400m", "vith14_laion", "vitl14_openai"], "full", "extra trees"),
    (["dinov2_l"], "full", "RBF SVM C=3"),
    (["dinov2_l", "eva02_l", "siglip_so400m", "vith14_laion", "vitl14_openai"], "full", "logreg C=10"),
]
SHORT = dict(zip(CLASSES, T.SHORT))


def main():
    orig = T.orig_idx
    y = T.y
    backbones = sorted({b for bs, _, _ in MODELS for b in bs})
    F = {b: T.load(b) for b in backbones}
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(y[orig], y[orig]))
    n = len(orig)
    rows = {}
    for bs, mode, clf in MODELS:
        name = f"{'+'.join(b.split('_')[0] for b in bs) if len(bs) < 5 else 'all'} | {clf}"
        wrong, pred_counts, p_true = np.zeros(n), np.zeros((n, 4)), np.zeros(n)
        for tr, va in splits:
            p = np.mean([T.fit_predict(F, b, mode, clf, orig[tr], "tr", orig[va]) for b in bs], 0)
            pred = p.argmax(1)
            wrong[va] += pred != y[orig[va]]
            pred_counts[va, pred] += 1
            p_true[va] += p[np.arange(len(va)), y[orig[va]]]
        rows[name] = (wrong / 10, pred_counts, p_true / 10)
        print(f"{name:40s} mean error {wrong.sum() / (10 * n):.3f}")

    df = T.train.iloc[orig][["id", "image_file", "label"]].reset_index(drop=True)
    for name, (err, counts, pt) in rows.items():
        df[f"err {name}"] = err
    df["err_mean"] = np.mean([r[0] for r in rows.values()], 0)
    allc = sum(r[1] for r in rows.values())
    df["usually_predicted"] = [CLASSES[k] for k in allc.argmax(1)]
    df["p_true_mean"] = np.mean([r[2] for r in rows.values()], 0)
    df = df.sort_values("err_mean", ascending=False)
    df.to_csv(LOG_DIR / "error_analysis.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[df.err_mean > 0].round(2).to_string(index=False))

    paths = image_paths()
    for chore in ("Washing Dishes", "Taking Out Trash"):
        sub = df[(df.err_mean >= 0.2) & df.label.str.startswith(chore)]
        if sub.empty:
            continue
        W, cols = 320, 4
        rows_n = (len(sub) + cols - 1) // cols
        sheet = Image.new("RGB", (W * cols, (W + 34) * rows_n), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(sub.itertuples()):
            im = ImageOps.exif_transpose(Image.open(paths[r.id])).convert("RGB")
            im.thumbnail((W - 6, W - 6))
            x, yy = (k % cols) * W, (k // cols) * (W + 34)
            sheet.paste(im, (x + 3, yy + 34))
            d.text((x + 4, yy + 2), f"#{r.id} true: {SHORT[r.label]}  wrong {r.err_mean:.0%}", fill="black")
            d.text((x + 4, yy + 16), f"usually predicted: {SHORT[r.usually_predicted]}", fill="red")
        out = LOG_DIR / f"error_sheet_{chore.split()[-1].lower()}.jpg"
        sheet.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
