"""Training losses (plan Eq. 6). Plan-sufficiency (Eq. 5) lands with the
planner module in stage C; reconstruction and rho regression are needed from
stage A/B and live here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension w.r.t. sorted errors (Berman et al.,
    CVPR'18, Alg. 1). gt_sorted: [P] binary, sorted by descending error."""
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
    return jaccard


def lovasz_binary(prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Lovász extension of the binary Jaccard loss on probabilities.
    prob/gt: flat [P] (already masked); errors = |gt - prob|."""
    if prob.numel() == 0:
        return prob.new_zeros(())
    errors = (gt - prob).abs()
    errors_sorted, order = torch.sort(errors, descending=True)
    return torch.dot(errors_sorted, _lovasz_grad(gt[order]))


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor,
                   ignore: int | None = None) -> torch.Tensor:
    """Multi-class Lovász-Softmax over classes PRESENT in the batch.
    probs: [P, C] softmax probabilities (masked); labels: [P] long.
    ignore: class id excluded from the class loop (GF-2: lovasz_ignore=17,
    the free class — its voxels still participate as background)."""
    if probs.numel() == 0:
        return probs.new_zeros(())
    losses = []
    for c in labels.unique():
        if ignore is not None and int(c) == ignore:
            continue
        fg = (labels == c).float()
        errors = (fg - probs[:, c]).abs()
        errors_sorted, order = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[order])))
    if not losses:
        return probs.new_zeros(())
    return torch.stack(losses).mean()


def official_occupancy_loss(occ_prob: torch.Tensor, sem_logit: torch.Tensor,
                            gt_label: torch.Tensor, visible: torch.Tensor,
                            class_weights: torch.Tensor | None = None,
                            ce_weight: float = 10.0, lovasz_weight: float = 1.0,
                            free_class: int = 17) -> torch.Tensor:
    """Stage-A loss = the GF-2 nuScenes recipe, verbatim where portable
    (encoder stage is no-innovation by discipline, 2026-07-16):
      weighted (C+1)-way voxel CE x10 + Lovász x1 with the free class
      excluded from the Lovász class loop, camera-visible voxels only
      (config/prob/nuscenes_gs6400.py:40-58: num_classes=18, empty_label=17,
      loss_voxel_ce_weight=10.0, loss_voxel_lovasz_weight=1.0,
      lovasz_ignore=17, manual_class_weight[18]).
    NOT ported: PixelDistributionLoss — it supervises the lifter's
    per-pixel depth-distribution initializer (pixel_logits/pixel_gt produced
    in gaussian_lifter_v2.py:169,200-201,320-321), a module we do not carry;
    it has no input in our architecture. Recorded, not substituted.

    The (C+1)-way distribution is assembled from the splat outputs:
      p(c) = occ_prob * softmax(sem)[c],  p(free) = 1 - occ_prob.
    class_weights: [C+1] tensor (last = free); None -> uniform. GF-2's
    manual_class_weight is fitted to SurroundOcc frequencies — recompute on
    Occ3D before full training (config-driven, scripts/train_encoder.py).
    """
    C = sem_logit.shape[-1]
    sem_prob = F.softmax(sem_logit, dim=-1)
    probs = torch.cat([occ_prob.unsqueeze(-1) * sem_prob,
                       (1.0 - occ_prob).unsqueeze(-1)], dim=-1)   # [B, V, C+1]
    logp = torch.log(probs.clamp_min(1e-6))

    gt = gt_label.clone()
    gt[gt == free_class] = C                    # free -> last channel
    m = visible.reshape(-1)
    ce = F.nll_loss(logp.reshape(-1, C + 1)[m], gt.reshape(-1)[m],
                    weight=class_weights)
    lov = lovasz_softmax(probs.reshape(-1, C + 1)[m], gt.reshape(-1)[m],
                         ignore=C)
    return ce_weight * ce + lovasz_weight * lov


def occupancy_recon_loss(occ_prob: torch.Tensor, sem_logit: torch.Tensor,
                         gt_label: torch.Tensor, visible: torch.Tensor,
                         free_class: int = 17,
                         voxel_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE on occupancy + CE on semantics of GT-occupied voxels + Lovász
    (binary Jaccard on occupancy + Lovász-Softmax on semantics — the GF-2
    training recipe uses weighted CE + Lovász too), camera-visible voxels
    only. voxel_weight [B, V] carries the plan-relevance weights w_i spread
    over each Gaussian's voxels (stage C; default ones).

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
        lov_sem = lovasz_softmax(F.softmax(sem_logit[sem_mask], dim=-1),
                                 gt_label[sem_mask])
    else:
        ce = occ_prob.new_zeros(())
        lov_sem = occ_prob.new_zeros(())

    lov_bin = lovasz_binary(occ_prob[visible], gt_occ[visible])

    return bce + ce + lov_bin + lov_sem


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
