"""Frozen pretrained image embeddings (image towers only) for every train / validation / test image.

WHAT THIS FILE DOES
    It passes every image (all 1177: train, validation, test) through large neural networks that were already
    trained by others on millions of images, and keeps each network's description of the image: one long vector
    of numbers called an EMBEDDING (768 to 2048 numbers per image, depending on the network).
    Images that look alike or show similar things get similar embeddings. A simple classifier trained on these
    vectors (baseline.py) then learns which embeddings mean "artifact" and which mean "clean".

WHY THIS IS NEEDED NEXT TO quality_features.py
    The hand-made quality numbers cannot tell WEATHER fog (clean) from LENS fog (artifact), or raindrops on a
    window seen through the glass (clean) from drops on the lens (artifact). That depends on what the image
    shows, and networks trained on huge photo collections capture that kind of meaning.
    Results (RBF SVM, CV F1): quality numbers alone 0.78; DINOv2 0.85; SigLIP 0.88; EVA02 0.88;
    all three + quality numbers 0.886 (run1).

"FROZEN" AND "IMAGE TOWER ONLY"
    Frozen: the networks' weights are never changed; they are only used to compute embeddings (no training here).
    Using fixed pretrained features with a simple classifier on top is a standard approach for small datasets [1][2].
    Image tower: CLIP-style models [3] have two parts, an image encoder and a text encoder, trained so that a
    photo and its caption get similar vectors. Only the image encoder is used; no text is involved anywhere.

THE NETWORKS (all are Vision Transformers, ViT [4])
    key            model                                    trained on                         input        numbers
    dinov2_l       DINOv2 ViT-L/14 [5]                      142M images, no labels             448 x 448    2048
    siglip_so400m  SigLIP, shape-optimised ViT (SO400M) [6][7]  WebLI image-text pairs          384 x 384    1152
    eva02_l        EVA02-CLIP ViT-L/14 [8][9]               2B image-text pairs (Merged-2B)    336 x 336     768
    vith14_laion   OpenCLIP ViT-H/14 [10]                   LAION-2B image-text pairs          (listed but never run;
                                                                                                no features file)
    DINOv2 learned from images alone; SigLIP and EVA02 learned from images with captions, so their embeddings
    encode concepts people write about ("foggy", "blurry", "through a window"). That is probably why they score
    higher here. "/14" means the network cuts the image into 14 x 14 pixel patches.

WHAT HAPPENS TO EACH IMAGE
    1. open, rotate upright (EXIF orientation, as in quality_features.py), convert to RGB
    2. shrink so the longer side is at most 1536 px (only to save time and memory; the model shrinks it further)
    3. make a mirrored (left-right flipped) copy
    4. each model's own preprocessing (see load_model) turns both copies into the exact input it expects:
         DINOv2, EVA02: resize the SHORT side to 448 / 336 px, then CUT a centred square. For a 4:3 photo the
                        outer 12.5% at each end of the long side is not seen by these two models.
         SigLIP:        SQUASH the whole image to 384 x 384 (nothing cut off, but the shape is distorted).
       then scale pixel values the way the model was trained (Normalize with the model's mean / std)
    5. the network turns each copy into one embedding vector
    6. every vector is scaled to length 1 ("L2-normalised"), so only its direction matters
    Both the normal and mirrored embedding are saved; baseline.py averages them. A mirrored photo is still the same
    photo, so averaging the two gives a slightly more stable description (flip averaging, as already used in
    AlexNet [11]).

OUTPUT (features/<key>.npz, one file per network)
    ids       (1177,)      image_id of each row
    split     (1177,)      "train", "validation" or "test"
    img       (1177, D)    embedding of the image as it is            (D = 2048, 1152 or 768)
    img_flip  (1177, D)    embedding of the mirrored image

HARDWARE / TIME
    Runs on the Apple-Silicon GPU ("mps") if available, else on the CPU. About 7-11 minutes per network for all
    1177 images on an M1 Pro. The network weights are downloaded once from Hugging Face and cached.

HOW THE FILE IS ORGANISED (read in this order)
    1. MODELS, settings   which networks, at which size
    2. unit()             scale vectors to length 1
    3. load_model()       load one network + its preprocessing; return an "encode" function
    4. extract()          run one network on all images and save the file
    5. main()             run the chosen networks one after another

REFERENCES
    [1] Kornblith, S., Shlens, J., Le, Q. V. (2019). Do better ImageNet models transfer better? Proc. CVPR 2019.
        (Fixed pretrained features + logistic regression as a transfer-learning baseline.)
    [2] Oquab, M., Bottou, L., Laptev, I., Sivic, J. (2014). Learning and transferring mid-level image
        representations using convolutional neural networks. Proc. CVPR 2014.
    [3] Radford, A. et al. (2021). Learning transferable visual models from natural language supervision (CLIP).
        Proc. ICML 2021. arXiv:2103.00020.
    [4] Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale.
        Proc. ICLR 2021. arXiv:2010.11929.
    [5] Oquab, M. et al. (2024). DINOv2: Learning robust visual features without supervision. Transactions on
        Machine Learning Research. arXiv:2304.07193.
    [6] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. (2023). Sigmoid loss for language image pre-training
        (SigLIP). Proc. ICCV 2023. arXiv:2303.15343.
    [7] Alabdulmohsin, I., Zhai, X., Kolesnikov, A., Beyer, L. (2023). Getting ViT in shape: Scaling laws for
        compute-optimal model design (SoViT-400m). Proc. NeurIPS 2023. arXiv:2305.13035.
    [8] Sun, Q., Fang, Y., Wu, L., Wang, X., Cao, Y. (2023). EVA-CLIP: Improved training techniques for CLIP at
        scale. arXiv:2303.15389.
    [9] Fang, Y. et al. (2024). EVA-02: A visual representation for neon genesis. Image and Vision Computing.
        arXiv:2303.11331.
    [10] Cherti, M. et al. (2023). Reproducible scaling laws for contrastive language-image learning (OpenCLIP).
        Proc. CVPR 2023. arXiv:2212.07143.
    [11] Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012). ImageNet classification with deep convolutional
        neural networks. Proc. NeurIPS 2012. (Averaging predictions over horizontal reflections at test time.)
    Software: open_clip (https://github.com/mlfoundations/open_clip) and timm, PyTorch Image Models
    (https://github.com/huggingface/pytorch-image-models), which download the pretrained weights.

Usage: python embed_features.py [model ...]   (default: dinov2_l siglip_so400m eva02_l)
       A model whose features file already exists is skipped (delete the file to recompute it).
"""
import sys                       # command-line arguments
import time                      # timing for the progress messages
from pathlib import Path         # (imported for consistency with the other scripts)

import numpy as np               # arrays; the embeddings are stored as numpy arrays
import torch                     # PyTorch, which runs the networks
from PIL import Image, ImageOps  # opening, rotating and mirroring images

# Shared with quality_features.py: the data / output folders and the list of all 1177 images, so every
# features file lists the images in exactly the same order.
from quality_features import DATA, OUT, all_images

Image.MAX_IMAGE_PIXELS = None    # allow very large images (trusted competition data; see quality_features.py)

# ----------------------------------------------------------------------------------------------------------
# 1. Which networks
# ----------------------------------------------------------------------------------------------------------
# key: (library that provides it, architecture name in that library, which trained weights)
#   timm      provides DINOv2; its weights are fixed by the architecture name (".lvd142m" = the LVD-142M
#             image collection it was trained on), so the third entry is None.
#   open_clip provides the CLIP-style models; the third entry names the training run of the weights
#             ("webli" = Google's WebLI image-text data; "merged2b_s6b_b61k" = 2B image-text pairs,
#             6B samples seen, batch size 61k).
MODELS = {
    "dinov2_l": ("timm", "vit_large_patch14_dinov2.lvd142m", None),
    "siglip_so400m": ("open_clip", "ViT-SO400M-14-SigLIP-384", "webli"),
    "eva02_l": ("open_clip", "EVA02-L-14-336", "merged2b_s6b_b61k"),
    "vith14_laion": ("open_clip", "ViT-H-14", "laion2b_s32b_b79k"),   # listed, but never run in this project
}
# DINOv2 input size. It must be a multiple of the 14-pixel patch size (448 = 32 x 14 -> a 32 x 32 grid of
# patches). Larger than DINOv2's usual 224 so that more fine detail (blur, drops) survives the resizing.
DINO_SIZE = 448
BATCH = 8        # images per batch; each image goes in twice (normal + mirrored), so 16 inputs per batch


# ----------------------------------------------------------------------------------------------------------
# 2. Scale vectors to length 1
# ----------------------------------------------------------------------------------------------------------
def unit(a):
    """Divide each vector (each row) by its length, so every vector has length 1 ("L2-normalisation").

    After this only the DIRECTION of a vector matters, not its size. Two images are then compared by the
    angle between their vectors (cosine similarity), which is how these networks are meant to be used.
    """
    # np.linalg.norm(..., axis=-1): the length of each row; keepdims=True keeps the shape (N, 1) so the
    # division works row by row.
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------------------------------------
# 3. Load one network
# ----------------------------------------------------------------------------------------------------------
def load_model(name, device):
    """Load one network on `device` and return (encode, preprocess).

    preprocess(PIL image) -> tensor of shape (3, S, S): the image resized / cropped / scaled exactly the way the
                             network was trained (S = 448, 384 or 336).
    encode(batch tensor)  -> numpy array (batch, D): one embedding vector per image.
    """
    lib, arch, pretrained = MODELS[name]
    if lib == "open_clip":
        import open_clip   # imported here so the library is only needed when one of its models is used
        # create_model_and_transforms returns (model, training transform, evaluation transform).
        # The training transform (random crops etc.) is not needed: "_" throws it away.
        # The evaluation transform "preprocess" is:
        #   EVA02:  Resize(short side -> 336) + CenterCrop(336 x 336) + Normalize
        #   SigLIP: Resize(whole image -> 384 x 384, squashed) + Normalize
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        # .to(device): move the network to the GPU (or CPU).
        # .eval(): switch off training-only behaviour (e.g. dropout), so the same image always gives the same
        # embedding.
        model = model.to(device).eval()
        # encode_image runs ONLY the image tower and returns its embedding.
        # .float() -> 32-bit numbers, .cpu() -> back from the GPU, .numpy() -> a numpy array.
        return (lambda x: model.encode_image(x).float().cpu().numpy()), preprocess

    # ---- otherwise: DINOv2 from timm ----
    import timm
    # num_classes=0: drop the classification head; we want the features, not class predictions.
    # img_size=DINO_SIZE: run the network at 448 x 448 instead of its default size.
    model = timm.create_model(arch, pretrained=True, num_classes=0, img_size=DINO_SIZE).to(device).eval()
    # Build the matching evaluation transform for a 448 x 448 input:
    #   Resize(short side -> 448) + CenterCrop(448 x 448) + Normalize with the ImageNet mean / std.
    cfg = timm.data.resolve_data_config({"input_size": (3, DINO_SIZE, DINO_SIZE)}, model=model)
    preprocess = timm.data.create_transform(**cfg)

    def encode(x):
        # forward_features returns one vector ("token") per position, shape (batch, 1 + 32*32, 1024):
        #   token 0          the CLS token: a summary of the whole image that the network builds up
        #   tokens 1..1024   one per 14 x 14 patch of the image (a 32 x 32 grid at 448 px)
        tok = model.forward_features(x).float().cpu().numpy()
        # The embedding is two 1024-number vectors side by side = 2048 numbers:
        #   unit(tok[:, 0])                                    the CLS token
        #   unit(tok[:, model.num_prefix_tokens:].mean(1))     the AVERAGE of all patch tokens
        # Both are common ways to summarise DINOv2's output [5]; the patch average keeps more information about
        # the whole frame (e.g. a smear in one area). num_prefix_tokens = how many tokens come before the
        # patch tokens (here 1: only CLS), so the slice starts at the first patch.
        # np.hstack puts the two vectors next to each other. Each is scaled to length 1 first so neither
        # dominates.
        return np.hstack([unit(tok[:, 0]), unit(tok[:, model.num_prefix_tokens:].mean(1))])
    return encode, preprocess


# ----------------------------------------------------------------------------------------------------------
# 4. Run one network on every image and save
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()   # no training here, so PyTorch does not need to remember intermediate values for
                   # gradients; this saves a lot of memory and time.
def extract(name, device):
    df = all_images()                                  # the 1177 images, in the shared order
    encode, preprocess = load_model(name, device)
    # feats: finished embeddings, one array per batch.  batch: preprocessed inputs waiting to be encoded.
    t0, feats, batch = time.time(), [], []

    def flush():
        """Encode everything waiting in `batch`, store the result, and empty the batch."""
        # torch.stack: list of (3, S, S) tensors -> one (n, 3, S, S) tensor; .to(device) moves it to the GPU.
        x = torch.stack(batch).to(device)
        # encode -> (n, D) embeddings; unit -> length 1 each. For DINOv2 this rescales the [CLS, patch-mean]
        # pair as a whole, so the full 2048-number vector has length 1.
        feats.append(unit(encode(x)))
        batch.clear()

    for k, fn in enumerate(df["filename"], 1):
        # Open, rotate upright by the EXIF tag, force RGB (same as load_rgb in quality_features.py).
        im = ImageOps.exif_transpose(Image.open(DATA / fn)).convert("RGB")
        # Shrink to at most 1536 px on the long side (aspect ratio kept). Only to save time and memory: the
        # model's own preprocessing shrinks it further to 336-448 px anyway.
        im.thumbnail((1536, 1536))
        # Add the image AND its mirror image (ImageOps.mirror flips left-right). The order in `batch` is
        # always normal, mirrored, normal, mirrored, ... which is used when saving below.
        batch += [preprocess(im), preprocess(ImageOps.mirror(im))]
        # When 8 images (16 inputs) are waiting, encode them together; GPUs are much faster on batches.
        if len(batch) >= BATCH * 2:
            flush()
        if k % 100 == 0 or k == len(df):               # progress, e.g. "[dinov2_l] 100/1177  58s"
            print(f"[{name}] {k}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    if batch:          # the last few images (1177 is not a multiple of 8)
        flush()

    # All embeddings in one array of shape (2 * 1177, D), alternating normal / mirrored.
    f = np.concatenate(feats)
    # f[0::2] = rows 0, 2, 4, ... = the normal images; f[1::2] = rows 1, 3, 5, ... = the mirrored ones.
    np.savez(OUT / f"{name}.npz", ids=df["image_id"].astype(str).values, split=df["split"].values,
             img=f[0::2], img_flip=f[1::2])
    print(f"[{name}] saved {f[0::2].shape} in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------------------
# 5. Run the chosen networks one after another
# ----------------------------------------------------------------------------------------------------------
def main(names):
    OUT.mkdir(exist_ok=True)                           # create features/ if needed
    # "mps" = the GPU of Apple-Silicon Macs (Metal Performance Shaders); fall back to the CPU otherwise.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for n in names:
        if (OUT / f"{n}.npz").exists():                # already computed: skip (delete the file to redo it)
            print(f"[{n}] exists, skipping")
            continue
        extract(n, device)
        if device == "mps":
            torch.mps.empty_cache()                    # free GPU memory before loading the next network


# Run directly: python embed_features.py [model ...]. With no model names, the three used in this project.
# When another script imports this file (tile_embed.py uses load_model and unit), nothing runs.
if __name__ == "__main__":
    main(sys.argv[1:] or ["dinov2_l", "siglip_so400m", "eva02_l"])
