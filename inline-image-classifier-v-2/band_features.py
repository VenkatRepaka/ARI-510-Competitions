"""Quality statistics on 3 horizontal bands (top / middle / bottom) and their differences.

Weather fog is depth dependent: far (top) bands are hazy, near (bottom) bands keep contrast and sharpness.
Lens fog / smears veil all bands evenly. Uses stats() from quality_features.py.
Output: features/bands.npz (ids, split, X, names) in the baseline.py format.
"""
import time

import numpy as np
import pandas as pd

from quality_features import DATA, OUT, all_images, load_rgb, stats

KEYS = ["lap_var", "grad_mean", "local_contrast", "dark_channel_mean", "bright_std", "sat_mean", "hf_ratio",
        "rms_contrast", "edge_density", "bright_mean"]


def features(path):
    rgb = load_rgb(path)
    h = rgb.shape[0]
    b = [stats(rgb[i * h // 3:(i + 1) * h // 3]) for i in range(3)]
    out = {}
    for k in KEYS:
        v = np.array([s[k] for s in b])
        out.update({f"{k}_top": v[0], f"{k}_mid": v[1], f"{k}_bot": v[2],
                    f"{k}_bot_minus_top": v[2] - v[0], f"{k}_band_std": v.std()})
    return out


def main():
    df = all_images()
    rows, t0 = [], time.time()
    for k, fn in enumerate(df["filename"], 1):
        rows.append(features(DATA / fn))
        if k % 200 == 0 or k == len(df):
            print(f"{k}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    f = pd.DataFrame(rows)
    np.savez(OUT / "bands.npz", ids=df["image_id"].astype(str).values, split=df["split"].values,
             X=f.values.astype(np.float32), names=np.array(f.columns))
    print("saved", f.shape)


if __name__ == "__main__":
    main()
