"""Save DINOv2 ViT-L/14 patch tokens (one vector per 14x14-pixel square) for every original photo.

The image is resized to 448x448 -> a 32x32 grid = 1024 patches, each a 1024-dim vector (L2-normalised, float16).
Saves features/dinov2_l_patches.npz with: ids (N,), patches (N, 1024, 1024), cls (N, 1024).
Web images (image_9NN_external.jpg) are skipped.

Usage: python extract_patches.py
"""
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from extract_features import DINO_SIZE, OUT, get_logger, image_paths, unit

log = get_logger("patches", "extract_patches.log")


@torch.no_grad()
def main():
    import timm
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              img_size=DINO_SIZE).to(device).eval()
    cfg = timm.data.resolve_data_config({"input_size": (3, DINO_SIZE, DINO_SIZE)}, model=model)
    # resize the whole image to 448x448 (no centre crop) so patches at the edges are kept
    import torchvision.transforms as T
    pre = T.Compose([T.Resize((DINO_SIZE, DINO_SIZE)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    paths = {i: p for i, p in image_paths().items() if "_external" not in p.name}
    ids = sorted(paths)
    P, C = [], []
    for k, i in enumerate(ids, 1):
        im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
        tok = model.forward_features(pre(im)[None].to(device)).float().cpu().numpy()[0]
        C.append(unit(tok[0]))
        P.append(unit(tok[model.num_prefix_tokens:]).astype(np.float16))
        if k % 20 == 0 or k == len(ids):
            log.info(f"image {k}/{len(ids)}  {time.time() - t0:.0f}s")
    np.savez(OUT / "dinov2_l_patches.npz", ids=np.array(ids), patches=np.array(P), cls=np.array(C))
    log.info(f"saved features/dinov2_l_patches.npz  patches {np.array(P).shape}")


if __name__ == "__main__":
    main()
