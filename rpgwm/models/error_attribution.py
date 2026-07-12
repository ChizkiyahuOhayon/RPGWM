"""M2 first half — measure how wrong each predicted Gaussian actually was
(plan §2.2, Eq. 2 inputs). Because slot identity is preserved through the
rollout, error is attributable per slot.

For every alive predicted Gaussian (already moved to the future-ego frame):
  support      = camera-visible voxels within 3σ of its center
  local_iou    = IoU(pred occupancy, GT occupancy) inside the support;
                 both empty -> 1.0 (correct emptiness)
  sem_err      = total-variation distance between the Gaussian's class
                 distribution and the GT class frequency in the support
                 (0 when the support has no GT-occupied voxel: the geometry
                 term already punishes hallucination)
  center_err   = ||mu - centroid(GT-occupied support)|| / mean std of the
                 Gaussian (dimensionless); 0 when no GT-occupied voxel
  has_gt_support = support non-empty; slots without it keep only the semantic
                 term and are EXCLUDED from calibration statistics.

Reference implementation, chunked over Gaussians; vectorized/CUDA later once
numerics are pinned by tests.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .gaussians import GaussianState

FREE_CLASS = 17


@torch.no_grad()
def per_gaussian_realized_error(state: GaussianState, occ_hard: torch.Tensor,
                                gt_label: torch.Tensor, visible: torch.Tensor,
                                voxel_centers: torch.Tensor, sigma_cut: float = 3.0,
                                chunk: int = 512):
    """One (batch of) predicted frame(s), already in the future-ego frame.

    state         predicted GaussianState [B, N, ...]
    occ_hard      [B, V] bool — the frame's own splatted hard occupancy
    gt_label      [B, V] long, 17 = free
    visible       [B, V] bool camera mask at the future timestamp
    voxel_centers [V, 3]

    Returns dict of [B, N] tensors: e (Eq. 2), local_iou, sem_err, center_err,
    has_gt_support (bool), valid (bool: alive & has_gt_support -> enters
    calibration and rho training), center_offset [B, N, 3] (meters, for the
    single-step noise fit; nan where undefined).
    """
    from .reliability import realized_error

    B, N = state.batch, state.n
    device = state.mu.device
    C = state.sem.shape[-1]
    sem_prob = F.softmax(state.sem, dim=-1)                      # [B, N, C]
    radius = sigma_cut * torch.exp(state.log_scale).max(-1).values  # [B, N]
    mean_std = torch.exp(state.log_scale).mean(-1)               # [B, N]
    gt_occ = gt_label != FREE_CLASS                              # [B, V]

    local_iou = torch.ones(B, N, device=device)
    sem_err = torch.zeros(B, N, device=device)
    center_err = torch.zeros(B, N, device=device)
    has_gt = torch.zeros(B, N, dtype=torch.bool, device=device)
    center_offset = torch.full((B, N, 3), float("nan"), device=device)

    for b in range(B):
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            # [n, V] support masks for this chunk
            d = torch.cdist(state.mu[b, s:e], voxel_centers)     # [n, V]
            support = (d <= radius[b, s:e, None]) & visible[b][None, :]
            has = support.any(-1)
            has_gt[b, s:e] = has

            p = occ_hard[b][None, :] & support                   # pred occ in support
            g = gt_occ[b][None, :] & support                     # gt occ in support
            inter = (p & g).sum(-1).float()
            union = (p | g).sum(-1).float()
            iou = torch.where(union > 0, inter / union.clamp_min(1.0),
                              torch.ones_like(union))            # both empty -> 1
            local_iou[b, s:e] = torch.where(has, iou, torch.ones_like(iou))

            # semantic + center terms need GT-occupied voxels in the support
            for j in range(e - s):
                gj = g[j]
                if not gj.any():
                    continue
                labels = gt_label[b][gj]                          # [m]
                freq = torch.bincount(labels, minlength=C)[:C].float()
                freq = freq / freq.sum().clamp_min(1.0)
                sem_err[b, s + j] = 0.5 * (sem_prob[b, s + j] - freq).abs().sum()
                centroid = voxel_centers[gj].mean(0)
                off = centroid - state.mu[b, s + j]
                center_offset[b, s + j] = off
                center_err[b, s + j] = off.norm() / mean_std[b, s + j].clamp_min(1e-3)

    e_val = realized_error(local_iou, sem_err, center_err, has_gt)
    valid = has_gt & state.alive_mask()
    return {"e": e_val, "local_iou": local_iou, "sem_err": sem_err,
            "center_err": center_err, "has_gt_support": has_gt,
            "valid": valid, "center_offset": center_offset}


def fit_single_step_noise(center_offsets_step0: torch.Tensor) -> torch.Tensor:
    """Per-axis variance of the measured one-step center offsets (meters),
    the noise term of the analytic covariance propagation (plan §2.2).
    center_offsets_step0: [M, 3] with nan rows for undefined slots."""
    ok = ~torch.isnan(center_offsets_step0).any(-1)
    if ok.sum() < 10:
        return torch.full((3,), 0.04)   # weak prior: 0.2 m std per axis
    return center_offsets_step0[ok].var(dim=0, unbiased=True).clamp_min(1e-6)
