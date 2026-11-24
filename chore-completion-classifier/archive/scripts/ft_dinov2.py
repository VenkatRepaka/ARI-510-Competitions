"""Fine-tune DINOv2-L end to end (last blocks) on the 4 chore classes, scored with the same photo-grouped CV.

Recipe (LP-FT, for tiny data):
  1. linear probe: backbone frozen, head (LayerNorm + Linear on CLS ++ mean-patch, 2048 -> 4) trained for LP_EPOCHS
  2. fine-tune: last UNFREEZE transformer blocks + final norm unfrozen, backbone lr BB_LR, head lr HEAD_LR,
     cosine decay, label smoothing 0.1, for FT_EPOCHS
  Training augmentation: random resized crop (35-100% of the area), mirror, colour jitter, at 448 px.
  Prediction: mean softmax of image + mirror (448 px, full photo, same as the frozen features).

Splits: the first --repeats repeats of aux_cv's 10x5 RepeatedStratifiedKFold over the 66 original photos;
external photos are only ever in training folds. The frozen DINOv2 RBF SVM is scored on the same folds so the
comparison is like for like.
-> features/ft_dinov2_oof.npz (out-of-fold probabilities), logs/ft_dinov2.log
Usage: python ft_dinov2.py [--repeats 1] [--folds 0 1 ...] [--epochs 12] [--unfreeze 4]
"""
import argparse
import math
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from PIL import ImageOps
from sklearn.metrics import f1_score
from torchvision import transforms as T

import aux_cv as A
from dish_crops import open_img
from extract_features import DINO_SIZE, OUT, get_logger, image_paths

log = get_logger("ft_dinov2", "ft_dinov2.log")
ARCH = "vit_large_patch14_dinov2.lvd142m"
LP_EPOCHS, HEAD_LR, BB_LR, BATCH = 3, 1e-3, 1e-5, 4      # batch 4: batch 8 swapped a 16 GB M1 Pro


class Net(nn.Module):
    def __init__(self, unfreeze):
        super().__init__()
        self.bb = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=DINO_SIZE)
        self.head = nn.Sequential(nn.LayerNorm(2048), nn.Dropout(0.3), nn.Linear(2048, 4))
        for p in self.bb.parameters():
            p.requires_grad = False
        self.tuned = list(self.bb.blocks[-unfreeze:]) + [self.bb.norm] if unfreeze else []
        self.n_frozen = len(self.bb.blocks) - unfreeze

    def forward(self, x, tune):
        bb = self.bb
        with torch.no_grad():                                      # frozen part: no activations kept
            x = bb.patch_embed(x); x = bb._pos_embed(x); x = bb.norm_pre(x)
            for blk in bb.blocks[:self.n_frozen]:
                x = blk(x)
            if not tune:
                for blk in bb.blocks[self.n_frozen:]:
                    x = blk(x)
                x = bb.norm(x)
        if tune:
            x = x.detach()
            for blk in bb.blocks[self.n_frozen:]:              # checkpointed: activations recomputed in backward
                x = checkpoint(blk, x, use_reentrant=False)
            x = bb.norm(x)
        feat = torch.cat([x[:, 0], x[:, bb.num_prefix_tokens:].mean(1)], 1)
        return self.head(feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--unfreeze", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "mps":
        torch.mps.set_per_process_memory_fraction(0.6)          # fail fast instead of swapping

    paths = image_paths()
    ids = list(A.train.id.values)
    imgs = {}
    for i in ids:
        im = open_img(int(i), paths); im.thumbnail((896, 896)); imgs[int(i)] = im
    cfg = timm.data.resolve_data_config({"input_size": (3, DINO_SIZE, DINO_SIZE)}, model=timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=DINO_SIZE))
    norm = T.Normalize(cfg["mean"], cfg["std"])
    train_tf = T.Compose([T.RandomResizedCrop(DINO_SIZE, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize((DINO_SIZE, DINO_SIZE)), T.ToTensor(), norm])

    splits = list(A.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(A.orig, A.y[A.orig]))
    todo = args.folds if args.folds is not None else list(range(5 * args.repeats))
    y = A.y
    oof_file = OUT / f"ft_dinov2{args.tag}_oof.npz"
    P = np.load(oof_file)["P"] if oof_file.exists() else np.full((10, len(y), 4), np.nan)

    for k in todo:
        t0 = time.time()
        tr, va = splits[k]
        tr_rows = np.r_[A.orig[tr], A.ext]; va_rows = A.orig[va]
        torch.manual_seed(k); np.random.seed(k)
        net = Net(args.unfreeze).to(dev)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
        total = LP_EPOCHS + args.epochs
        steps_per = math.ceil(len(tr_rows) / BATCH)
        opt = torch.optim.AdamW([{"params": net.head.parameters(), "lr": HEAD_LR, "weight_decay": 1e-4}], lr=HEAD_LR)
        step = 0
        for ep in range(total):
            tune = ep >= LP_EPOCHS
            if ep == LP_EPOCHS and net.tuned:                     # switch to fine-tuning
                for m in net.tuned:
                    for p in m.parameters():
                        p.requires_grad = True
                opt = torch.optim.AdamW([{"params": net.head.parameters(), "lr": HEAD_LR / 10},
                                         {"params": [p for m in net.tuned for p in m.parameters()], "lr": BB_LR}],
                                        weight_decay=0.05)
                base_lrs = [g["lr"] for g in opt.param_groups]; step = 0
            net.train()
            order = np.random.permutation(tr_rows); losses = []
            for s in range(0, len(order), BATCH):
                b = order[s:s + BATCH]
                x = torch.stack([train_tf(imgs[int(ids[r])]) for r in b]).to(dev)
                t = torch.tensor(y[b]).to(dev)
                if tune:
                    f = 0.5 * (1 + math.cos(math.pi * step / (args.epochs * steps_per)))
                    for g, lr in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr * f
                loss = lossf(net(x, tune), t)
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                losses.append(loss.item())
                if dev == "mps" and step % 5 == 0:
                    torch.mps.empty_cache()
            log.info(f"fold {k} ep {ep + 1}/{total} {'ft' if tune else 'lp'} loss {np.mean(losses):.3f}")
        net.eval()
        with torch.no_grad():
            for r in va_rows:
                im = imgs[int(ids[r])]
                x = torch.stack([eval_tf(im), eval_tf(ImageOps.mirror(im))]).to(dev)
                P[k // 5, r] = net(x, False).softmax(1).mean(0).cpu().numpy()
        np.savez(oof_file, P=P)
        acc = (P[k // 5, va_rows].argmax(1) == y[va_rows]).mean()
        wrong = [int(ids[r]) for r in va_rows if P[k // 5, r].argmax() != y[r]]
        log.info(f"fold {k} done in {(time.time() - t0) / 60:.1f} min  acc {acc:.3f}  wrong {wrong}")
        del net, opt
        if dev == "mps":
            torch.mps.empty_cache()

    # summary over complete repeats, next to the frozen SVM on the same folds
    for r in range(10):
        if np.isnan(P[r, A.orig]).any():
            continue
        Pf = np.zeros((len(y), 4))
        for tr, va in splits[r * 5:(r + 1) * 5]:
            X = A.BASE["dinov2 full"]; trr = np.r_[A.orig[tr], A.ext]
            Pf[A.orig[va]] = A.proba(A.MODELS["RBF SVM"]().fit(X[trr], y[trr]), X[A.orig[va]])
        o = A.orig
        for name, Q in (("fine-tuned", P[r]), ("frozen SVM", Pf), ("average", (P[r] + Pf) / 2)):
            pred = Q[o].argmax(1); d = np.isin(y[o], [0, 1])
            wrong = sorted(int(ids[j]) for j in o[pred != y[o]])
            log.info(f"repeat {r} {name:11s} F1 {f1_score(y[o], pred, average='macro'):.3f}  "
                     f"dishes F1 {f1_score(y[o][d], pred[d], average='macro', labels=[0, 1]):.3f}  wrong {wrong}")


if __name__ == "__main__":
    main()
