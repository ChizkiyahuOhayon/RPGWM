"""Gaussian scene state: the single object that lives from perception through
rollout to the planner. Slot identity is positional — index i at frame t is the
same physical slot at frame t+k. Slots are never inserted, deleted, or permuted;
they only die/revive through the opacity gate.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

OPACITY_ALIVE_THRESHOLD = 0.05


class GaussianState:
    """Batched container of N Gaussian slots.

    Tensors (all [B, N, ...]):
      mu        [B, N, 3]   center, ego frame of the *current* timestamp (meters)
      log_scale [B, N, 3]   log of per-axis std (meters)
      quat      [B, N, 4]   unit quaternion (w, x, y, z)
      opacity   [B, N]      in [0, 1]; slot alive iff opacity > 0.05
      sem       [B, N, C]   semantic logits over C classes
      feat      [B, N, D]   working feature carried across modules
    """

    FIELDS = ("mu", "log_scale", "quat", "opacity", "sem", "feat")

    def __init__(self, mu, log_scale, quat, opacity, sem, feat):
        self.mu = mu
        self.log_scale = log_scale
        self.quat = quat
        self.opacity = opacity
        self.sem = sem
        self.feat = feat

    # -- constructors ------------------------------------------------------
    @staticmethod
    def random(batch: int, n: int, num_classes: int, feat_dim: int,
               extent: float = 40.0, device: str = "cpu",
               generator: torch.Generator | None = None) -> "GaussianState":
        g = generator
        mu = (torch.rand(batch, n, 3, device=device, generator=g) - 0.5) * 2 * extent
        mu[..., 2] = mu[..., 2] * 0.05  # keep near ground plane
        log_scale = torch.randn(batch, n, 3, device=device, generator=g) * 0.2 - 0.5
        quat = F.normalize(torch.randn(batch, n, 4, device=device, generator=g), dim=-1)
        opacity = torch.rand(batch, n, device=device, generator=g)
        sem = torch.randn(batch, n, num_classes, device=device, generator=g)
        feat = torch.randn(batch, n, feat_dim, device=device, generator=g)
        return GaussianState(mu, log_scale, quat, opacity, sem, feat)

    # -- properties --------------------------------------------------------
    @property
    def batch(self) -> int:
        return self.mu.shape[0]

    @property
    def n(self) -> int:
        return self.mu.shape[1]

    def alive_mask(self) -> torch.Tensor:
        """[B, N] bool — the opacity gate. Dead slots keep flowing through
        modules (identity must be preserved) but are masked downstream."""
        return self.opacity > OPACITY_ALIVE_THRESHOLD

    def covariance(self) -> torch.Tensor:
        """[B, N, 3, 3] Sigma = R diag(exp(2*log_scale)) R^T."""
        return quat_to_rotmat(self.quat) @ torch.diag_embed(
            torch.exp(2.0 * self.log_scale)) @ quat_to_rotmat(self.quat).transpose(-1, -2)

    def detach(self) -> "GaussianState":
        return GaussianState(*(getattr(self, f).detach() for f in self.FIELDS))

    def clone(self) -> "GaussianState":
        return GaussianState(*(getattr(self, f).clone() for f in self.FIELDS))


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(w,x,y,z) unit quaternion -> rotation matrix. q: [..., 4] -> [..., 3, 3]."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1)
    row1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1)
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return torch.stack([row0, row1, row2], -2)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ∘ q2, both [..., 4] (w,x,y,z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], -1)
