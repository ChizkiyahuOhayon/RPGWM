"""Training losses (plan Eq. 6). Plan-sufficiency (Eq. 5) lands with the
planner module in stage C; reconstruction and rho regression are needed from
stage A/B and live here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def occupancy_recon_loss(occ_prob: torch.Tensor, sem_logit: torch.Tensor,
                         gt_label: torch.Tensor, visible: torch.Tensor,
                         free_class: int = 17,
                         voxel_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE on occupancy + CE on semantics of GT-occupied voxels + soft-IoU,
    camera-visible voxels only. voxel_weight [B, V] carries the plan-relevance
    weights w_i spread over each Gaussian's voxels (stage C; default ones).

    occ_prob [B, V] in [0,1); sem_logit [B, V, C]; gt_label [B, V] long;
    visible [B, V] bool.
    """
    gt_occ = (gt_label != free_class).float()
    w = voxel_weight if voxel_weight is not None else torch.ones_like(occ_prob)
    m = visible.float() * w

    bce = F.binary_cross_entropy(occ_prob.clamp(1e-6, 1 - 1e-6), gt_occ, weight=m,
                                 reduction="sum") / m.sum().clamp_min(1.0)

    sem_mask = visible & (gt_label != free_class)
    if sem_mask.any():
        ce = F.cross_entropy(sem_logit[sem_mask], gt_label[sem_mask],
                             reduction="mean")
    else:
        ce = occ_prob.new_zeros(())

    # soft-IoU (Lovász stand-in until the server run pins the full loss)
    inter = (occ_prob * gt_occ * visible.float()).sum()
    union = ((occ_prob + gt_occ - occ_prob * gt_occ) * visible.float()).sum()
    soft_iou = 1.0 - inter / union.clamp_min(1.0)

    return bce + ce + soft_iou


def rho_regression_loss(rho: torch.Tensor, target_q: torch.Tensor,
                        valid: torch.Tensor) -> torch.Tensor:
    """MSE between predicted unreliability and the quantile-normalized realized
    error, over slots that have GT support (plan §2.2)."""
    diff = (rho - target_q) ** 2 * valid.float()
    return diff.sum() / valid.float().sum().clamp_min(1.0)


def plan_sufficiency_loss(traj_pred: torch.Tensor, traj_ref: torch.Tensor,
                          mode_logit_pred: torch.Tensor,
                          mode_logit_ref: torch.Tensor) -> torch.Tensor:
    """Eq. 5: ||pi(G_hat) - pi(G_gt)||² + KL(p_mode(G_gt) || p_mode(G_hat)).
    The reference branch must be detached by the caller (stop-gradient)."""
    l2 = ((traj_pred - traj_ref) ** 2).sum(-1).mean()
    kl = F.kl_div(F.log_softmax(mode_logit_pred, dim=-1),
                  F.softmax(mode_logit_ref, dim=-1), reduction="batchmean")
    return l2 + kl
