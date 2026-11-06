"""
Train the arm pose generator (learning/arm_model.py).

    python -m learning.arm_train --dof 4

Each training example is one pose from one reference path, together with that
problem's obstacle image, start and goal. The start and goal poses themselves
are left out (the planner already has them), so each solved problem gives
N_POSES - 2 examples.

Loss = reconstruction error + KL term (the standard VAE objective). The
reconstruction term treats the decoder as a Gaussian with standard deviation
SIGMA (in units of pi rad); smaller SIGMA = sharper poses, larger = more
spread. The KL term keeps the latent codes close to N(0, 1) so that random
noise decodes to sensible poses at planning time.

Augmentation: mirror the world top-to-bottom through the arm's base. A
mirrored problem has every joint angle negated (start, goal and path), and
the obstacle image flipped -- an exact symmetry, so it doubles the data for
free. (Rotations are NOT exact symmetries here: rotating the world turns the
first joint, which can push it outside its [-pi, pi] limit.)

Validation metric, chosen to be easy to interpret: for each validation
problem, draw 64 poses from the network and 64 uniformly at random, and
measure each pose's distance (joint space, radians) to the reference path.
"closer" = mean uniform distance / mean generated distance. 1.0 means the
network is no better than random; higher is better.

Which epoch to keep? The validation loss and the "closer" metric disagree
(the loss bottoms out early while "closer" keeps improving), and neither is
the thing that matters: whether the PLANNER improves. So training saves a few
checkpoints, and `arm_evaluate --tune` picks checkpoint and mix together by
planning on validation worlds. The chosen one is copied to
learning/models/arm<dof>_cvae.pt.

Outputs:
    learning/models/arm<dof>_cvae_e<epoch>.pt   one per checkpoint epoch
    results/arm/training_log_arm<dof>.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .arm_data import load_split
from .arm_model import ArmCVAE, save_model

SIGMA = 0.1     # decoder std, in units of pi radians (~0.31 rad)


def examples(split):
    """Flatten solved problems into (problem index, pose) training pairs."""
    keep = np.flatnonzero(split["solved"])
    poses = split["poses"][keep, 1:-1, :]                    # drop start and goal
    idx = np.repeat(keep, poses.shape[1])
    return idx, poses.reshape(-1, poses.shape[-1])


def batch_tensors(split, idx, q, mirror):
    occ = torch.from_numpy(split["occ"][idx].astype(np.float32))
    s = torch.from_numpy(split["start"][idx].astype(np.float32))
    g = torch.from_numpy(split["goal"][idx].astype(np.float32))
    q = torch.from_numpy(q.astype(np.float32))
    if mirror is not None:
        # Flip row order of the image (y -> -y about the base) and negate angles.
        m = torch.from_numpy(mirror)
        occ[m] = torch.flip(occ[m], dims=(1,))
        s[m], g[m], q[m] = -s[m], -g[m], -q[m]
    return occ, s, g, q


def vae_loss(recon, q, mu, logvar):
    rec = (((recon - q) / math.pi) ** 2).sum(dim=1) / (2 * SIGMA ** 2)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)
    return (rec + kl).mean(), rec.mean().item(), kl.mean().item()


def point_to_path(points, path):
    """Joint-space distance from each point (M, d) to a polyline path (K, d)."""
    A, B = path[:-1], path[1:]                               # (K-1, d)
    AB = B - A
    AP = points[:, None, :] - A[None]                        # (M, K-1, d)
    denom = np.maximum((AB * AB).sum(-1), 1e-12)
    t = np.clip((AP * AB[None]).sum(-1) / denom, 0, 1)
    closest = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


@torch.no_grad()
def closeness(model, split, n_problems=100, n=64, seed=0):
    """Mean uniform distance / mean generated distance to the reference path."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    ratios = []
    for i in np.flatnonzero(split["solved"])[:n_problems]:
        q = model.generate(torch.from_numpy(split["occ"][i].astype(np.float32)),
                           torch.from_numpy(split["start"][i].astype(np.float32)),
                           torch.from_numpy(split["goal"][i].astype(np.float32)),
                           n, generator=gen).numpy()
        u = rng.uniform(-math.pi, math.pi, size=(n, split["dof"]))
        path = split["poses"][i]
        ratios.append(point_to_path(u, path).mean() / point_to_path(q, path).mean())
    return float(np.mean(ratios))


def evaluate_loss(model, split, idx, q):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for k in range(0, len(idx), 1024):
            occ, s, g, qq = batch_tensors(split, idx[k:k + 1024], q[k:k + 1024], None)
            recon, mu, logvar = model(occ, s, g, qq)
            total += vae_loss(recon, qq, mu, logvar)[0].item() * len(qq)
    return total / len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dof", type=int, default=4)
    ap.add_argument("--data", default=os.path.join("learning", "data"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoints", default="10,30,60",
                    help="epochs at which to save the model; tuning picks among them")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    train = load_split(os.path.join(args.data, "arm%d_train.npz" % args.dof))
    val = load_split(os.path.join(args.data, "arm%d_val.npz" % args.dof))
    t_idx, t_q = examples(train)
    v_idx, v_q = examples(val)

    checkpoints = {int(e) for e in args.checkpoints.split(",")}
    model = ArmCVAE(args.dof, grid=train["occ"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = os.path.join("learning", "models", "arm%d_cvae.pt" % args.dof)
    log_path = os.path.join("results", "arm", "training_log_arm%d.json" % args.dof)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print("dof %d | train %d poses from %d solved problems | val %d poses | %d params"
          % (args.dof, len(t_idx), train["solved"].sum(), len(v_idx), n_params))
    log = {"args": vars(args), "sigma": SIGMA, "n_train_poses": len(t_idx),
           "n_train_problems": int(train["solved"].sum()), "params": n_params,
           "epochs": []}
    best = math.inf

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(len(t_idx))
        run_rec = run_kl = 0.0
        for k in range(0, len(perm), args.batch):
            b = perm[k:k + args.batch]
            mirror = rng.random(len(b)) < 0.5
            occ, s, g, q = batch_tensors(train, t_idx[b], t_q[b], mirror)
            recon, mu, logvar = model(occ, s, g, q)
            loss, rec, kl = vae_loss(recon, q, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_rec += rec * len(b)
            run_kl += kl * len(b)

        val_loss = evaluate_loss(model, val, v_idx, v_q)
        row = {"epoch": epoch, "train_rec": run_rec / len(perm), "train_kl": run_kl / len(perm),
               "val_loss": val_loss, "seconds": time.perf_counter() - t0}
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            row["val_closer"] = closeness(model, val)
        log["epochs"].append(row)
        mark = ""
        best = min(best, val_loss)
        if epoch in checkpoints:
            path = out.replace(".pt", "_e%d.pt" % epoch)
            save_model(model, path, {"epoch": epoch, "val_loss": val_loss,
                                     "val_closer": row.get("val_closer")})
            mark = "  * checkpoint"
        print("epoch %2d  rec %.2f  kl %.2f  val %.2f%s  (%.0f s)%s" % (
            epoch, row["train_rec"], row["train_kl"], val_loss,
            ("  closer x%.2f" % row["val_closer"]) if "val_closer" in row else "",
            row["seconds"], mark), flush=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    print("lowest val loss %.3f; checkpoints at epochs %s. Run "
          "`python -m learning.arm_evaluate --tune` to choose one by planning results."
          % (best, sorted(checkpoints)))


if __name__ == "__main__":
    main()
