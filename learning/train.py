"""
Train the sampling network.

    python -m learning.train                       # defaults below
    python -m learning.train --epochs 5 --quick    # smoke test

Loss: binary cross-entropy per cell, computed only over *free* cells. Blocked
cells are excluded because the input already says they are blocked -- counting
them would reward the network for copying channel 0, inflating the score
without teaching it anything about paths.

Augmentation: each training map is randomly rotated by 0/90/180/270 degrees
and optionally mirrored (the 8 symmetries of a square). A path problem
rotated is still a path problem with the label rotated the same way, so this
multiplies the effective data by 8 for free and stops the network preferring
any particular direction.

Model selection uses the validation split only. The test problems are never
touched here.

Outputs:
    learning/models/sampler_unet.pt       weights of the best epoch (val loss)
    results/learned/training_log.json     per-epoch losses and val IoU
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import load_split
from .features import GRID, make_features
from .model import SamplingUNet, count_parameters, save_model


def _cell(p):
    """World (x, y) -> continuous grid coords. Mirrors features.to_cell_coords
    for the project's 100 x 100 workspace, vectorised over a batch."""
    return p / 100.0 * GRID - 0.5


def build_tensors(split):
    """Precompute the network inputs for a whole split.

    Stored at half precision to halve memory (10k maps = ~400 MB); converted
    back to float32 one batch at a time.
    """
    n = len(split["free"])
    X = np.empty((n, 5, GRID, GRID), dtype=np.float16)
    s = _cell(split["start"])
    g = _cell(split["goal"])
    for i in range(n):
        X[i] = make_features(split["free"][i], s[i], g[i])
    Y = split["label"].astype(np.float16)
    M = split["free"].astype(np.float16)
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M)


def augment(x, y, m, gen):
    """Apply one random square symmetry to the whole batch."""
    k = int(torch.randint(4, (1,), generator=gen))
    flip = bool(torch.randint(2, (1,), generator=gen))
    x, y, m = (torch.rot90(t, k, dims=(-2, -1)) for t in (x, y, m))
    if flip:
        x, y, m = (torch.flip(t, dims=(-1,)) for t in (x, y, m))
    return x, y, m


def masked_bce(logits, y, m, pos_weight):
    loss = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=pos_weight)
    return (loss * m).sum() / m.sum()


@torch.no_grad()
def evaluate(model, X, Y, M, pos_weight, batch=256):
    model.eval()
    total, inter, union, n = 0.0, 0.0, 0.0, 0
    for i in range(0, len(X), batch):
        logits = model(X[i:i + batch].float())
        y, m = Y[i:i + batch].float(), M[i:i + batch].float()
        total += masked_bce(logits, y, m, pos_weight).item() * len(y)
        pred = (logits > 0).float() * m
        inter += (pred * y).sum().item()
        union += ((pred + y) > 0).float().mul(m).sum().item()
        n += len(y)
    return total / n, inter / max(union, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join("learning", "data"))
    ap.add_argument("--out", default=os.path.join("learning", "models", "sampler_unet.pt"))
    ap.add_argument("--log", default=os.path.join("results", "learned", "training_log.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="use 1000 training maps")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)

    train = load_split(os.path.join(args.data, "train.npz"))
    val = load_split(os.path.join(args.data, "val.npz"))
    Xt, Yt, Mt = build_tensors(train)
    Xv, Yv, Mv = build_tensors(val)
    if args.quick:
        Xt, Yt, Mt = Xt[:1000], Yt[:1000], Mt[:1000]

    # Positive cells are a minority (~15% of free space); weighting them by
    # the class ratio stops the network collapsing to "predict nothing".
    # (summed in float64: a float16 sum over millions of cells overflows)
    pos_frac = (Yt.double() * Mt.double()).sum() / Mt.double().sum()
    pos_weight = ((1 - pos_frac) / pos_frac).clamp(max=10.0).float()

    model = SamplingUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(Xt) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    print("train %d  val %d  params %d  pos_frac %.3f  pos_weight %.2f"
          % (len(Xt), len(Xv), count_parameters(model), pos_frac, pos_weight))

    log = {"args": vars(args), "n_train": len(Xt), "n_val": len(Xv),
           "params": count_parameters(model), "pos_weight": float(pos_weight),
           "epochs": []}
    best = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        running, seen = 0.0, 0
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i + args.batch]
            x, y, m = augment(Xt[idx].float(), Yt[idx].float(), Mt[idx].float(), gen)
            loss = masked_bce(model(x), y, m, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(idx)
            seen += len(idx)

        val_loss, val_iou = evaluate(model, Xv, Yv, Mv, pos_weight)
        row = {"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss,
               "val_iou": val_iou, "seconds": time.perf_counter() - t0}
        log["epochs"].append(row)
        marker = ""
        if val_loss < best:
            best = val_loss
            save_model(model, args.out, {"epoch": epoch, "val_loss": val_loss,
                                         "val_iou": val_iou})
            marker = "  * saved"
        print("epoch %2d  train %.4f  val %.4f  IoU %.3f  (%.0f s)%s"
              % (epoch, row["train_loss"], val_loss, val_iou, row["seconds"], marker),
              flush=True)
        with open(args.log, "w") as f:
            json.dump(log, f, indent=2)

    print("best val loss %.4f -> %s" % (best, args.out))


if __name__ == "__main__":
    main()
