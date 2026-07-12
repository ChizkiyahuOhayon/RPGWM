"""M2 — per-Gaussian unreliability rho (plan §2.2, Eq. 2).

Three pieces:
  1. realized_error: measured rollout error e_i per predicted Gaussian
     (local splatted-IoU deficit in its 3σ support + attribute error),
  2. QuantileNormalizer: maps raw e_i to [0,1] by its rank within the
     training-set error distribution of the SAME future step
     (coherence fix: direction = UNreliability, larger = less trusted),
  3. RhoHead: small MLP predicting Q(e_i) from the slot's rollout feature,
     an analytically propagated positional-covariance summary
     (the BeliefGauss legacy), and a step embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 1. realized error
# --------------------------------------------------------------------------
def realized_error(local_iou: torch.Tensor, sem_err: torch.Tensor,
                   center_err: torch.Tensor, has_gt_support: torch.Tensor) -> torch.Tensor:
    """e_i = (1 - local IoU) + ||Δattr||, per plan Eq. 2.

    local_iou      [B, N] splatted-IoU of the Gaussian's 3σ neighborhood vs GT
                   (camera-visible voxels only)
    sem_err        [B, N] semantic-probability error vs local GT class freq
    center_err     [B, N] center vs GT-occupancy centroid, scale-normalized
    has_gt_support [B, N] bool — False -> keep only the semantic term and
                   exclude from calibration statistics (handled by caller)
    """
    attr = torch.sqrt(sem_err ** 2 + center_err ** 2 + 1e-12)
    e = (1.0 - local_iou) + attr
    return torch.where(has_gt_support, e, sem_err)


# --------------------------------------------------------------------------
# 2. quantile normalization (per future step, fitted on the training set)
# --------------------------------------------------------------------------
class QuantileNormalizer:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.sorted_errors: list[torch.Tensor | None] = [None] * num_steps

    def fit(self, step: int, errors: torch.Tensor) -> None:
        self.sorted_errors[step] = torch.sort(errors.flatten().detach()).values

    def transform(self, step: int, errors: torch.Tensor) -> torch.Tensor:
        ref = self.sorted_errors[step]
        assert ref is not None, f"QuantileNormalizer not fitted for step {step}"
        rank = torch.searchsorted(ref, errors.flatten().contiguous())
        q = rank.float() / max(len(ref), 1)
        return q.clamp(0.0, 1.0).view_as(errors)

    def state_dict(self):
        return {"sorted_errors": self.sorted_errors}

    def load_state_dict(self, sd):
        self.sorted_errors = sd["sorted_errors"]


# --------------------------------------------------------------------------
# 3. analytically propagated covariance summary (BeliefGauss legacy)
# --------------------------------------------------------------------------
def propagated_cov_summary(step: int, single_step_pos_var: torch.Tensor) -> torch.Tensor:
    """Closed-form positional covariance after `step+1` rollout steps under the
    fitted isotropic single-step noise model: P_k = (k+1) * diag(var).

    single_step_pos_var: [3] per-axis position-residual variance fitted on
    measured one-step errors (stage B).
    Returns [5]: log eigenvalues (diag here), log trace, log det.
    """
    p = (step + 1) * single_step_pos_var
    return torch.cat([torch.log(p + 1e-9),
                      torch.log(p.sum() + 1e-9).view(1),
                      torch.log(p.prod() + 1e-9).view(1)])


class RhoHead(nn.Module):
    """rho_i = h(f_i, Sigma_prop, step) ≈ Q(e_i) ∈ [0, 1]. 3-layer MLP, width 256."""

    COV_DIM = 5

    def __init__(self, feat_dim: int = 256, num_steps: int = 6, width: int = 256):
        super().__init__()
        self.step_embed = nn.Embedding(num_steps, 16)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + self.COV_DIM + 16, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, feat: torch.Tensor, cov_summary: torch.Tensor,
                step: torch.Tensor) -> torch.Tensor:
        """feat [B, N, D]; cov_summary [B, N, 5] (or [5], broadcast);
        step [B, N] long. Returns rho [B, N] in [0, 1]."""
        if cov_summary.dim() == 1:
            cov_summary = cov_summary.expand(*feat.shape[:2], -1)
        x = torch.cat([feat, cov_summary, self.step_embed(step)], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


# --------------------------------------------------------------------------
# calibration check (ECE over 10 equal-mass bins)
# --------------------------------------------------------------------------
def expected_calibration_error(rho: torch.Tensor, target_q: torch.Tensor,
                               bins: int = 10) -> torch.Tensor:
    """Weighted |mean(rho) - mean(target)| over equal-mass bins of rho."""
    rho, target_q = rho.flatten(), target_q.flatten()
    order = torch.argsort(rho)
    rho, target_q = rho[order], target_q[order]
    splits = torch.tensor_split(torch.arange(len(rho)), bins)
    ece = rho.new_zeros(())
    for idx in splits:
        if len(idx) == 0:
            continue
        ece = ece + (len(idx) / len(rho)) * (rho[idx].mean() - target_q[idx].mean()).abs()
    return ece


# --------------------------------------------------------------------------
# partition conditioning helpers (plan Eq. 4)
# --------------------------------------------------------------------------
def partition_and_inflate(log_scale: torch.Tensor, rho: torch.Tensor,
                          theta: float, lam: float):
    """Split by rho > theta; inflate untrusted covariance by (1 + lam*rho).
    Sigma scales with exp(2*log_scale), so inflating Sigma by f multiplies
    log_scale by 0.5*log(f). Returns (trusted_mask [B,N], new_log_scale)."""
    trusted = rho <= theta
    factor = 1.0 + lam * rho
    inflated = log_scale + 0.5 * torch.log(factor).unsqueeze(-1)
    new_log_scale = torch.where(trusted.unsqueeze(-1), log_scale, inflated)
    return trusted, new_log_scale
