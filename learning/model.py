"""
The network: a small U-Net that maps a problem image to a per-cell score.

Input  (B, 5, 64, 64)  -- see features.py
Output (B, 64, 64)     -- logits; sigmoid(logit) = predicted probability that
                          the cell lies on a near-optimal path (see labels.py)

Why a U-Net (Ronneberger et al., 2015)
--------------------------------------
The answer at one cell depends on the *whole* map: whether a cell is worth
sampling depends on where the walls are between it and the goal. The
down-sampling path shrinks the map 64 -> 4 so the deepest layer sees the
entire world at once; the up-sampling path brings that global decision back
to full resolution, and the skip connections keep the exact obstacle edges
sharp. This is the same architecture family as Neural RRT* (Wang et al.,
IEEE T-ASE 2020), kept small (~0.9 M parameters) so it trains on a laptop CPU
and runs in a few milliseconds at planning time.

Imports PyTorch; nothing under planners/ or env/ does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_CHANNELS


class _Block(nn.Module):
    """Two 3x3 convolutions, each followed by batch-norm and ReLU."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SamplingUNet(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, widths=(16, 32, 64, 96, 128)):
        super().__init__()
        self.widths = tuple(widths)
        self.down = nn.ModuleList()
        c = in_channels
        for w in widths:
            self.down.append(_Block(c, w))
            c = w
        self.up = nn.ModuleList()
        for w_skip in reversed(widths[:-1]):
            self.up.append(_Block(c + w_skip, w_skip))
            c = w_skip
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = block(torch.cat([x, skips.pop()], dim=1))
        return self.head(x).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_model(model, path, extra=None):
    payload = {"state_dict": model.state_dict(), "widths": model.widths}
    payload.update(extra or {})
    torch.save(payload, path)


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SamplingUNet(widths=payload["widths"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
