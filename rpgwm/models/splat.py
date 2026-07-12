"""Differentiable Gaussian -> voxel occupancy splatting (plan §2.1, Eq. after (1)).

Reference PyTorch implementation, chunked over voxels so it runs on CPU for
tests and on GPU without a custom kernel. Semantics of the plan:
  - every voxel center within 3σ of a Gaussian receives contribution
    opacity * exp(-0.5 * Mahalanobis²),
  - voxel is occupied when the summed contribution exceeds tau (default 0.5),
  - class label = contribution-weighted vote over the contributing Gaussians'
    semantic logits.
The transform is differentiable; the forecast loss backpropagates through it.
A fused CUDA kernel (S2GO-style blocked splatting) replaces this on the server
once numerics are pinned by these tests.
"""
from __future__ import annotations

import torch

from .gaussians import GaussianState


class VoxelGrid:
    """Axis-aligned grid in the ego frame at the scored timestamp."""

    def __init__(self, xyz_min=(-40.0, -40.0, -1.0), xyz_max=(40.0, 40.0, 5.4),
                 resolution=(200, 200, 16)):
        self.xyz_min = torch.tensor(xyz_min)
        self.xyz_max = torch.tensor(xyz_max)
        self.resolution = tuple(resolution)

    def centers(self, device="cpu") -> torch.Tensor:
        """[V, 3] voxel centers, V = X*Y*Z (row-major x, y, z)."""
        lo, hi = self.xyz_min.to(device), self.xyz_max.to(device)
        axes = [lo[i] + (hi[i] - lo[i]) * (torch.arange(r, device=device) + 0.5) / r
                for i, r in enumerate(self.resolution)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def splat_occupancy(state: GaussianState, grid: VoxelGrid, tau: float = 0.5,
                    chunk: int = 65536, sigma_cut: float = 3.0):
    """Splat one Gaussian frame to the grid.

    Returns:
      occ_prob  [B, V]     soft occupancy in [0, 1) via 1 - exp(-sum contrib)
      occ_hard  [B, V]     bool, summed contribution > tau (non-differentiable)
      sem_logit [B, V, C]  contribution-weighted semantic logits
    """
    device = state.mu.device
    B, N, C = state.batch, state.n, state.sem.shape[-1]
    centers = grid.centers(device)                              # [V, 3]
    V = centers.shape[0]

    cov = state.covariance()                                    # [B, N, 3, 3]
    prec = torch.linalg.inv(cov + 1e-6 * torch.eye(3, device=device))
    alive = state.alive_mask().float()                          # [B, N]
    op = state.opacity * alive                                  # dead slots contribute 0
    # 3σ cutoff radius per Gaussian (isotropic bound: 3 * largest std)
    radius = sigma_cut * torch.exp(state.log_scale).max(-1).values  # [B, N]

    contrib_sum = torch.zeros(B, V, device=device)
    sem_acc = torch.zeros(B, V, C, device=device)
    for s in range(0, V, chunk):
        pts = centers[s:s + chunk]                              # [v, 3]
        diff = pts.view(1, 1, -1, 3) - state.mu.unsqueeze(2)    # [B, N, v, 3]
        maha = torch.einsum("bnvi,bnij,bnvj->bnv", diff, prec, diff)
        w = op.unsqueeze(-1) * torch.exp(-0.5 * maha)           # [B, N, v]
        # hard 3σ box cut keeps the reference implementation consistent
        # with the future CUDA kernel's neighbor list
        w = w * (diff.norm(dim=-1) <= radius.unsqueeze(-1)).float()
        contrib_sum[:, s:s + chunk] = w.sum(1)
        sem_acc[:, s:s + chunk] = torch.einsum("bnv,bnc->bvc", w, state.sem)

    occ_prob = 1.0 - torch.exp(-contrib_sum)
    occ_hard = contrib_sum > tau
    sem_logit = sem_acc / (contrib_sum.unsqueeze(-1) + 1e-6)
    return occ_prob, occ_hard, sem_logit


def transform_to_future_ego(state: GaussianState, rot: torch.Tensor,
                            trans: torch.Tensor) -> GaussianState:
    """Move Gaussians from current-ego frame into the future-ego frame before
    scoring (benchmark protocol). rot: [B, 3, 3], trans: [B, 3] mapping
    current-ego coords -> future-ego coords."""
    from .gaussians import quat_multiply
    import torch.nn.functional as F
    mu = torch.einsum("bij,bnj->bni", rot, state.mu) + trans.unsqueeze(1)
    # rotation as quaternion (w,x,y,z) from rot — use trace-based conversion
    q = rotmat_to_quat(rot)                                     # [B, 4]
    quat = F.normalize(quat_multiply(q.unsqueeze(1).expand_as(state.quat), state.quat), dim=-1)
    return GaussianState(mu, state.log_scale, quat, state.opacity, state.sem, state.feat)


def rotmat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """[B, 3, 3] -> [B, 4] (w,x,y,z). Stable for proper rotations."""
    B = m.shape[0]
    w = torch.sqrt((1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]).clamp_min(1e-8)) / 2
    x = (m[:, 2, 1] - m[:, 1, 2]) / (4 * w)
    y = (m[:, 0, 2] - m[:, 2, 0]) / (4 * w)
    z = (m[:, 1, 0] - m[:, 0, 1]) / (4 * w)
    return torch.stack([w, x, y, z], dim=-1)
