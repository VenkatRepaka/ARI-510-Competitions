"""Summary features from 3x3 tile embeddings (features/tiles_<model>.npz), saved in the baseline.py format (ids, X).

tilesim_<m>.npz : ~20 scalars on how alike the tiles are (lens fog / smears make every tile look alike;
                  weather fog leaves the near, bottom tiles different from the far, top ones):
                  pairwise tile cosine mean/min/std, top-row vs bottom-row, centre vs border, left vs right,
                  each tile vs the full-image embedding (mean/min/bottom row/top row), adjacent-tile similarity.
tileagg_<m>.npz : tilesim + mean tile embedding + per-dimension std over tiles + (top row - bottom row) mean embedding.
Usage: python tile_features.py [model ...]
"""
import sys

import numpy as np

from quality_features import OUT

TOP, MID, BOT = [0, 1, 2], [3, 4, 5], [6, 7, 8]
LEFT, RIGHT, BORDER = [0, 3, 6], [2, 5, 8], [0, 1, 2, 3, 5, 6, 7, 8]


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def sim_features(T, full):
    """T (N, 9, D) unit tiles, full (N, D) unit full-image embedding -> (N, F), names."""
    S = np.einsum("nid,njd->nij", T, T)
    iu = np.triu_indices(9, 1)
    pair = S[:, iu[0], iu[1]]
    tf = np.einsum("nid,nd->ni", T, full)
    rowmean = lambda idx: unit(T[:, idx].mean(1))
    cos = lambda a, b: (rowmean(a) * rowmean(b)).sum(1)
    adj = [(r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2)] + [(i, i + 3) for i in range(6)]
    vert = [(i, i + 3) for i in range(6)]
    f = {
        "pair_mean": pair.mean(1), "pair_min": pair.min(1), "pair_std": pair.std(1),
        "adj_mean": np.mean([S[:, a, b] for a, b in adj], 0),
        "vert_adj_mean": np.mean([S[:, a, b] for a, b in vert], 0),
        "top_vs_bottom": cos(TOP, BOT), "top_vs_mid": cos(TOP, MID), "mid_vs_bottom": cos(MID, BOT),
        "left_vs_right": cos(LEFT, RIGHT), "centre_vs_border": (T[:, 4] * rowmean(BORDER)).sum(1),
        "tile_full_mean": tf.mean(1), "tile_full_min": tf.min(1), "tile_full_std": tf.std(1),
        "tile_full_top": tf[:, TOP].mean(1), "tile_full_bottom": tf[:, BOT].mean(1),
        "tile_full_centre": tf[:, 4], "tile_full_border": tf[:, BORDER].mean(1),
        "tile_full_bottom_minus_top": tf[:, BOT].mean(1) - tf[:, TOP].mean(1),
        "tile_full_centre_minus_border": tf[:, 4] - tf[:, BORDER].mean(1),
    }
    return np.column_stack(list(f.values())), list(f)


def main(models):
    for m in models:
        z = np.load(OUT / f"tiles_{m}.npz", allow_pickle=True)
        full = np.load(OUT / f"{m}.npz", allow_pickle=True)
        assert (z["ids"] == full["ids"]).all()
        T = unit(z["tiles"].astype(np.float32))
        sims, names = sim_features(T, unit(full["img"].astype(np.float32)))
        np.savez(OUT / f"tilesim_{m}.npz", ids=z["ids"], split=z["split"], X=sims.astype(np.float32),
                 names=np.array(names))
        agg = np.hstack([sims, unit(T.mean(1)), T.std(1), T[:, TOP].mean(1) - T[:, BOT].mean(1)])
        np.savez(OUT / f"tileagg_{m}.npz", ids=z["ids"], split=z["split"], X=agg.astype(np.float32))
        print(m, "tilesim", sims.shape, "tileagg", agg.shape)


if __name__ == "__main__":
    main(sys.argv[1:] or ["eva02_l", "siglip_so400m"])
