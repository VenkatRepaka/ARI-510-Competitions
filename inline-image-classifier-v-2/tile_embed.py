"""Embed a 3x3 grid of tiles per image (row-major: tile 0 = top-left, 4 = centre, 8 = bottom-right).

Idea: lens fog / smears / droplets on the lens veil the whole frame evenly, near objects included, while
weather fog thins towards the camera (bottom of the frame stays sharp and contrasty). Comparing tiles can tell these apart.
Tiles overlap slightly (each is 40% of width/height, centres on a 3x3 grid) so objects on tile borders are kept.

Output: features/tiles_<model>.npz with ids, split, tiles (N, 9, D) float16, L2-normalised.
Usage: python tile_embed.py [model ...]   (default: eva02_l siglip_so400m)
"""
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from embed_features import load_model, unit
from quality_features import DATA, OUT, all_images

Image.MAX_IMAGE_PIXELS = None
FRAC = 0.4


def tiles(im):
    w, h = im.size
    tw, th = int(w * FRAC), int(h * FRAC)
    out = []
    for r in range(3):
        for c in range(3):
            x, y = int((w - tw) * c / 2), int((h - th) * r / 2)
            out.append(im.crop((x, y, x + tw, y + th)))
    return out


@torch.no_grad()
def extract(name, device):
    df = all_images()
    encode, preprocess = load_model(name, device)
    t0, feats = time.time(), []
    for k, fn in enumerate(df["filename"], 1):
        im = ImageOps.exif_transpose(Image.open(DATA / fn)).convert("RGB")
        im.thumbnail((2048, 2048))
        x = torch.stack([preprocess(t) for t in tiles(im)]).to(device)
        feats.append(unit(encode(x)).astype(np.float16))
        if k % 100 == 0 or k == len(df):
            el = time.time() - t0
            print(f"[{name}] {k}/{len(df)}  {el:.0f}s  eta {el / k * (len(df) - k):.0f}s", flush=True)
    np.savez(OUT / f"tiles_{name}.npz", ids=df["image_id"].astype(str).values, split=df["split"].values,
             tiles=np.stack(feats))
    print(f"[{name}] saved {np.stack(feats).shape}", flush=True)


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for n in sys.argv[1:] or ["eva02_l", "siglip_so400m"]:
        if (OUT / f"tiles_{n}.npz").exists():
            print(f"[{n}] exists, skipping")
            continue
        extract(n, device)
        if device == "mps":
            torch.mps.empty_cache()
