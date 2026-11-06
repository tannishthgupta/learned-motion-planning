"""
A conditional VAE that proposes arm poses for a given planning problem.

Why not a heatmap like the 2-D robot?
-------------------------------------
For the 2-D robot the network scored every cell of a 64 x 64 floor plan. For
an arm with N joints the equivalent would be a grid over N angles (64^4 = 16
million cells for 4 joints, 64^7 = 4 trillion for 7). So instead of scoring
every possible pose, this network *generates* poses directly: give it random
noise and it returns a pose that looks like the poses on good paths for this
particular problem. That scales to any number of joints.

How a conditional VAE works (Ichter, Harrison & Pavone, ICRA 2018)
------------------------------------------------------------------
    condition c   = what we know about the problem: obstacle image + start
                    and goal angles, squeezed into a vector by `ConditionNet`
    latent z      = a few random numbers (here LATENT = 6)

    encoder  (pose q, c) -> a Gaussian over z     [used only in training]
    decoder  (z, c)      -> a pose q              [used for sampling]

Training shows it poses from reference paths. The encoder learns to compress
each pose into z, the decoder learns to rebuild it, and a KL penalty keeps the
z values looking like plain N(0, 1) noise. At planning time the encoder is
thrown away: draw z ~ N(0, 1), run the decoder, and out comes a plausible
pose. Different z give different poses, so the network can represent
"go round the left OR the right" instead of averaging the two into a pose
that runs straight through the obstacle.

Angles are scaled to [-1, 1] (divided by pi) inside the network.

Imports PyTorch; nothing under arm/, env/ or planners/ does.
"""

import math

import torch
import torch.nn as nn

LATENT = 6


class ConditionNet(nn.Module):
    """Obstacle image + start/goal angles -> one condition vector."""

    def __init__(self, dof, grid=32, out=128):
        super().__init__()
        # Three stride-2 convolutions: 32 -> 16 -> 8 -> 4, so the last layer
        # sees the whole world.
        self.image = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (grid // 8) ** 2, 128), nn.ReLU(),
        )
        self.endpoints = nn.Sequential(nn.Linear(2 * dof, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(128 + 64, out), nn.ReLU())

    def forward(self, occ, start, goal):
        """occ (B, grid, grid) in {0,1}; start, goal (B, dof) in radians."""
        img = self.image(occ.unsqueeze(1))
        ends = self.endpoints(torch.cat([start, goal], dim=1) / math.pi)
        return self.mix(torch.cat([img, ends], dim=1))


def _mlp(n_in, n_out, hidden=256):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


class ArmCVAE(nn.Module):
    def __init__(self, dof, grid=32, cond_dim=128, latent=LATENT):
        super().__init__()
        self.dof, self.grid, self.latent = dof, grid, latent
        self.condition = ConditionNet(dof, grid, cond_dim)
        self.encoder = _mlp(dof + cond_dim, 2 * latent)
        self.decoder = _mlp(latent + cond_dim, dof)

    def decode(self, z, c):
        """Latent + condition -> pose in radians, kept inside the joint limits."""
        return torch.tanh(self.decoder(torch.cat([z, c], dim=1))) * math.pi

    def forward(self, occ, start, goal, q):
        """Training pass: returns (reconstructed pose, mu, logvar)."""
        c = self.condition(occ, start, goal)
        stats = self.encoder(torch.cat([q / math.pi, c], dim=1))
        mu, logvar = stats[:, :self.latent], stats[:, self.latent:]
        # Reparameterisation trick: sample z in a way gradients can flow through.
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, c), mu, logvar

    @torch.no_grad()
    def generate(self, occ, start, goal, n, generator=None):
        """n poses for ONE problem. occ (grid, grid); start/goal (dof,)."""
        c = self.condition(occ[None], start[None], goal[None]).expand(n, -1)
        z = torch.randn(n, self.latent, generator=generator)
        return self.decode(z, c)


def save_model(model, path, extra=None):
    payload = {"state_dict": model.state_dict(), "dof": model.dof, "grid": model.grid,
               "latent": model.latent}
    payload.update(extra or {})
    torch.save(payload, path)


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ArmCVAE(payload["dof"], payload["grid"], latent=payload["latent"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload
