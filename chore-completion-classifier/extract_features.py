"""Extract image embeddings (no text encoder) for every photo with several pretrained backbones.

For each photo we encode 7 views: full image, its mirror, and 5 crops (4 corners at 2/3 size + centre).
Saves features/<model>.npz with: ids, img (N,D), img_flip (N,D), crops (N,5,D). All vectors are L2-normalised.

With --aug N it instead encodes N random augmentations per photo (random crop, flip, colour jitter)
and saves features/<model>_aug.npz with: ids, aug (N_photos, N, D).
With --color it re-encodes the 7 views under each colour change in COLOR_NAMES (applied to the whole photo first)
and saves features/<model>_color.npz with: ids, color (N_photos, len(COLOR_NAMES), 7, D).

Photos already in an existing .npz are not re-encoded: only new photos (e.g. image_901_external.jpg) are added.

Usage: python extract_features.py [model ...]               (default: all models)
       python extract_features.py --force dinov2_l          (re-extract)
       python extract_features.py --aug 20 dinov2_l eva02_l (augmented views)
       python extract_features.py --color dinov2_l          (colour-changed views)
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "data" / "images"
OUT = ROOT / "features"
LOG_DIR = ROOT / "logs"

CLASSES = [
    "Washing Dishes|Complete",
    "Washing Dishes|Incomplete",
    "Taking Out Trash|Complete",
    "Taking Out Trash|Incomplete",
]

# name: (library, architecture, pretrained weights). Only the image towers of the CLIP models are used.
MODELS = {
    "vitl14_openai": ("open_clip", "ViT-L-14", "openai"),
    "vith14_laion": ("open_clip", "ViT-H-14", "laion2b_s32b_b79k"),
    "siglip_so400m": ("open_clip", "ViT-SO400M-14-SigLIP-384", "webli"),
    "eva02_l": ("open_clip", "EVA02-L-14-336", "merged2b_s6b_b61k"),
    "dinov2_l": ("timm", "vit_large_patch14_dinov2.lvd142m", None),
}
DINO_SIZE = 448  # multiple of the 14-pixel patch size


def get_logger(name, log_file):
    LOG_DIR.mkdir(exist_ok=True)
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / log_file)):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


log = get_logger("extract", "extract_features.log")


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def views(im):
    w, h = im.size
    cw, ch = int(w * 2 / 3), int(h * 2 / 3)
    boxes = [(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch), ((w - cw) // 2, (h - ch) // 2)]
    return [im, ImageOps.mirror(im)] + [im.crop((x, y, x + cw, y + ch)) for x, y in boxes]


def _scale_rgb(im, factors):
    return Image.merge("RGB", [c.point(lambda p, f=f: min(255, int(p * f))) for c, f in zip(im.split(), factors)])


# Colour changes for --color. The unchanged photo is not repeated: it is already in features/<model>.npz.
COLOR_CHANGES = {
    "brighter (gamma 0.6)": lambda im: _gamma(im, 0.6),
    "darker (gamma 1.6)": lambda im: _gamma(im, 1.6),
    "warm (R x1.15, B x0.85)": lambda im: _scale_rgb(im, (1.15, 1.0, 0.85)),
    "cool (R x0.85, B x1.15)": lambda im: _scale_rgb(im, (0.85, 1.0, 1.15)),
    "autocontrast": lambda im: ImageOps.autocontrast(im, cutoff=1),
}
COLOR_NAMES = list(COLOR_CHANGES)


def _gamma(im, g):
    return im.point(lambda p: int(255 * (p / 255) ** g + 0.5))


def color_views(im):
    """The 7 views of every colour-changed copy of the photo: len(COLOR_NAMES) * 7 images."""
    return [v for change in COLOR_CHANGES.values() for v in views(change(im))]


def image_paths():
    """{id: path} for image_NNN.jpg and image_NNN_<suffix>.jpg (e.g. image_901_external.jpg)."""
    return {int(p.stem.split("_")[1]): p for p in IMG_DIR.glob("image_*.jpg")}


def load_model(name, device):
    lib, arch, pretrained = MODELS[name]
    if lib == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = model.to(device).eval()
        encode = lambda x: model.encode_image(x).float().cpu().numpy()
    else:
        import timm
        model = timm.create_model(arch, pretrained=True, num_classes=0, img_size=DINO_SIZE).to(device).eval()
        cfg = timm.data.resolve_data_config({"input_size": (3, DINO_SIZE, DINO_SIZE)}, model=model)
        preprocess = timm.data.create_transform(**cfg)

        def encode(x):  # CLS token next to the mean patch token
            tokens = model.forward_features(x).float().cpu().numpy()
            return np.hstack([unit(tokens[:, 0]), unit(tokens[:, model.num_prefix_tokens:].mean(1))])
    return encode, preprocess


def augmented_views(im, n, seed):
    """n random views: crop 30-100% of the area, random mirror, mild colour jitter. Seeded per photo."""
    from torchvision import transforms as T
    torch.manual_seed(seed)
    jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)
    out = []
    for _ in range(n):
        top, left, h, w = T.RandomResizedCrop.get_params(im, scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3))
        v = im.crop((left, top, left + w, top + h))
        if torch.rand(1).item() < 0.5:
            v = ImageOps.mirror(v)
        out.append(jitter(v))
    return out


@torch.no_grad()
def extract(name, device, n_aug=0, force=False, color=False):
    t0 = time.time()
    out_name = f"{name}_color" if color else (f"{name}_aug" if n_aug else name)
    log.info(f"[{out_name}] loading {MODELS[name][1]} on {device}")
    encode, preprocess = load_model(name, device)
    log.info(f"[{out_name}] loaded in {time.time() - t0:.0f}s")
    paths = image_paths()
    out_path = OUT / f"{out_name}.npz"
    old = dict(np.load(out_path)) if out_path.exists() and not force else None
    ids = sorted(set(paths) - set(old["ids"])) if old is not None else sorted(paths)
    log.info(f"[{out_name}] {len(ids)} photos to encode" + (" (missing from existing file)" if old else ""))
    feats = []
    for k, i in enumerate(ids, 1):
        im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
        im.thumbnail((1536, 1536))
        vs = color_views(im) if color else (augmented_views(im, n_aug, seed=i) if n_aug else views(im))
        feats.append(unit(encode(torch.stack([preprocess(v) for v in vs]).to(device))))
        if k % 10 == 0 or k == len(ids):
            el = time.time() - t0
            log.info(f"[{out_name}] image {k}/{len(ids)}  elapsed {el:.0f}s  eta {el / k * (len(ids) - k):.0f}s")
    feats = np.array(feats)
    if color:
        new = dict(ids=np.array(ids), color=feats.reshape(len(ids), len(COLOR_NAMES), 7, -1))
    elif n_aug:
        new = dict(ids=np.array(ids), aug=feats)
    else:
        new = \
        dict(ids=np.array(ids), img=feats[:, 0], img_flip=feats[:, 1], crops=feats[:, 2:])
    if old is not None:  # append the newly encoded photos to the existing file
        new = {k: np.concatenate([old[k], new[k]]) for k in new}
    np.savez(out_path, **new)
    log.info(f"[{out_name}] saved features/{out_name}.npz  shape {feats.shape}  total {time.time() - t0:.0f}s")


def main(names, force=False, n_aug=0, color=False):
    OUT.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    for k, name in enumerate(names, 1):
        out_name = f"{name}_color" if color else (f"{name}_aug" if n_aug else name)
        log.info(f"===== model {k}/{len(names)}: {out_name} =====")
        if (OUT / f"{out_name}.npz").exists() and not force:
            have = set(np.load(OUT / f"{out_name}.npz")["ids"])
            if set(image_paths()) <= have:
                log.info(f"[{out_name}] already on disk with every photo, skipping (use --force to redo)")
                continue
        try:
            extract(name, device, n_aug, force, color)
        except Exception:
            log.exception(f"[{out_name}] FAILED")
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    n_aug = 0
    if "--aug" in args:
        k = args.index("--aug")
        n_aug = int(args[k + 1])
        del args[k:k + 2]
    color = "--color" in args
    args = [a for a in args if a != "--color"]
    main(args or list(MODELS), force, n_aug, color)
