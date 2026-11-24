"""Foreground-pooled DINOv2 features: average only the patches that belong to the main object.

1. Patch tokens: features/dinov2_l_patches.npz holds the 66 originals + 36 test photos (extract_patches.py).
   The labelled external photos in data/train.csv are encoded here the same way into
   features/dinov2_l_patches_external.npz.
2. Foreground mask (see foreground()): per photo, k-means on its patch tokens; clusters that crowd the image border
   are background, the rest is the object. Falls back to the centre box when no object stands out.
3. Saves features/dinov2_l_fg.npz with: ids, fg (N, 1024) = unit mean of foreground patches,
   fg_frac (N,) = share of patches marked foreground, fallback (N,) bool, mask (N, 32, 32) bool.
   Also writes logs/fg_maps.jpg with the masks drawn over selected photos.

Usage: python fg_features.py [--maps 63 87 100 2 ...]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps

from extract_features import DINO_SIZE, IMG_DIR, LOG_DIR, OUT, get_logger, image_paths, unit

log = get_logger("fg", "fg_features.log")
GRID = DINO_SIZE // 14  # 32


@torch.no_grad()
def extract_external_patches():
    """Patch tokens for labelled external photos missing from the patch files."""
    out_path = OUT / "dinov2_l_patches_external.npz"
    train = pd.read_csv(IMG_DIR.parent / "train.csv")
    want = set(train.id[train.image_file.str.contains("_external")])
    have = set(np.load(out_path)["ids"]) if out_path.exists() else set()
    todo = sorted(want - have)
    if not todo:
        log.info("external patches already on disk")
        return
    import timm
    import torchvision.transforms as T
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              img_size=DINO_SIZE).to(device).eval()
    cfg = timm.data.resolve_data_config({"input_size": (3, DINO_SIZE, DINO_SIZE)}, model=model)
    pre = T.Compose([T.Resize((DINO_SIZE, DINO_SIZE)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    paths = image_paths()
    P, C = [], []
    t0 = time.time()
    for i in todo:
        im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
        tok = model.forward_features(pre(im)[None].to(device)).float().cpu().numpy()[0]
        C.append(unit(tok[0]))
        P.append(unit(tok[model.num_prefix_tokens:]).astype(np.float16))
    new = dict(ids=np.array(todo), patches=np.array(P), cls=np.array(C))
    if have:
        old = dict(np.load(out_path))
        new = {k: np.concatenate([old[k], new[k]]) for k in new}
    np.savez(out_path, **new)
    log.info(f"encoded {len(todo)} external photos in {time.time() - t0:.0f}s -> {out_path.name}")


def load_patches():
    parts = [np.load(OUT / "dinov2_l_patches.npz")]
    if (OUT / "dinov2_l_patches_external.npz").exists():
        parts.append(np.load(OUT / "dinov2_l_patches_external.npz"))
    ids = np.concatenate([p["ids"] for p in parts])
    patches = np.concatenate([p["patches"] for p in parts]).astype(np.float32)
    return ids, patches


def foreground(patches):
    """Per photo: cluster its patches (k = 2, 3, 4), call a cluster background when it is over-represented on the
    2-patch image border, and keep the clustering whose foreground (5-60% of the photo) sits most in the centre.
    When no clustering qualifies, the centre box (middle 2/3) is used. Returns (mask (N, 1024) bool, used_fallback (N,))."""
    from sklearn.cluster import KMeans
    grid = np.zeros((GRID, GRID), bool)
    grid[:2], grid[-2:], grid[:, :2], grid[:, -2:] = True, True, True, True
    border = grid.ravel()
    yy, xx = np.mgrid[:GRID, :GRID]
    centre = ((np.abs(yy - GRID / 2 + .5) < GRID / 3) & (np.abs(xx - GRID / 2 + .5) < GRID / 3)).ravel()
    masks, fallback = [], []
    for p in patches:
        best = None
        for k in (2, 3, 4):
            lab = KMeans(k, n_init=4, random_state=0).fit_predict(p)
            fg_clusters = [c for c in range(k) if (lab[border] == c).mean() < 0.5 * (lab == c).mean()]
            m = np.isin(lab, fg_clusters)
            if 0.05 <= m.mean() <= 0.6:
                score = m[centre].mean() - m[border].mean()
                if best is None or score > best[0]:
                    best = (score, m)
        fallback.append(best is None)
        masks.append(centre.copy() if best is None else best[1])
    return np.array(masks), np.array(fallback)


def draw_maps(ids, mask, show, path):
    S = 220
    paths = image_paths()
    pos = {i: k for k, i in enumerate(ids)}
    tiles = []
    for i in show:
        im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB").resize((S, S))
        m = Image.fromarray((mask[pos[i]].reshape(GRID, GRID) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
        dim = Image.blend(im, Image.new("RGB", (S, S), "black"), 0.65)
        over = Image.composite(im, dim, m)                       # foreground bright, background dimmed
        t = Image.new("RGB", (2 * S, S + 18), "white")
        t.paste(im, (0, 0)); t.paste(over, (S, 0))
        ImageDraw.Draw(t).text((4, S + 3), f"#{i}  foreground {mask[pos[i]].mean():.0%}", fill="black")
        tiles.append(t)
    cols = 3
    sheet = Image.new("RGB", (cols * 2 * S, -(-len(tiles) // cols) * (S + 18)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * 2 * S, (k // cols) * (S + 18)))
    sheet.save(path, quality=88)


def main(show):
    extract_external_patches()
    ids, patches = load_patches()
    mask, fallback = foreground(patches)
    fg = np.array([unit(p[m].mean(0)) for p, m in zip(patches, mask)])
    np.savez(OUT / "dinov2_l_fg.npz", ids=ids, fg=fg, fg_frac=mask.mean(1), fallback=fallback,
             mask=mask.reshape(-1, GRID, GRID))
    log.info(f"saved features/dinov2_l_fg.npz for {len(ids)} photos; foreground share median "
             f"{np.median(mask.mean(1)):.0%}; centre-box fallback for {fallback.sum()} photos")
    draw_maps(ids, mask, show, LOG_DIR / "fg_maps.jpg")
    log.info(f"wrote logs/fg_maps.jpg for {show}")


if __name__ == "__main__":
    args = sys.argv[1:]
    show = [int(a) for a in args[args.index("--maps") + 1:]] if "--maps" in args else [63, 87, 100, 2, 41, 12, 72, 19, 80]
    main(show)
