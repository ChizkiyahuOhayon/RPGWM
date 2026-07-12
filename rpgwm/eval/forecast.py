"""Occ3D-nuScenes 4D forecasting protocol (plan Table 1).

Masked binary IoU (occupied vs free) and per-class mIoU at each future step,
counting only camera-visible voxels — the OccWorld-lineage protocol. This
module must be cross-checked against the official OccWorld eval numbers on the
server before any headline number is trusted (see SERVER_RUNBOOK.md step 3).
"""
from __future__ import annotations

import torch

FREE_CLASS = 17  # Occ3D: 0..16 semantics, 17 = free


def binary_iou(pred_occ: torch.Tensor, gt_occ: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """pred_occ/gt_occ/mask: [...] bool, same shape. IoU over masked voxels."""
    p, g = pred_occ & mask, gt_occ & mask
    inter = (p & g).sum().float()
    union = (p | g).sum().float()
    return inter / union.clamp_min(1.0)


def semantic_miou(pred_label: torch.Tensor, gt_label: torch.Tensor,
                  mask: torch.Tensor, num_classes: int = 17):
    """Per-class IoU over masked voxels, ignoring the free class in both;
    classes absent from GT and prediction are excluded from the mean.
    pred_label/gt_label: [...] long in [0, num_classes] (num_classes==free).
    Returns (miou scalar, per_class [num_classes] with nan for absent)."""
    per_class = torch.full((num_classes,), float("nan"))
    valid = mask & (gt_label != FREE_CLASS)
    for c in range(num_classes):
        p = (pred_label == c) & valid
        g = (gt_label == c) & valid
        union = (p | g).sum()
        if union == 0:
            continue
        per_class[c] = ((p & g).sum().float() / union.float()).item()
    present = ~torch.isnan(per_class)
    miou = per_class[present].mean() if present.any() else torch.tensor(0.0)
    return miou, per_class


def forecast_scores(pred_frames, gt_frames, masks, num_classes: int = 17):
    """Score a K-step rollout.

    pred_frames: list of (occ_hard [V] bool, sem_label [V] long)
    gt_frames:   list of gt_label [V] long (free = FREE_CLASS)
    masks:       list of visibility masks [V] bool (future-timestamp camera mask)
    Returns dict {step: {"iou": float, "miou": float}} plus averages.
    """
    out, ious, mious = {}, [], []
    for k, ((occ, sem), gt, m) in enumerate(zip(pred_frames, gt_frames, masks)):
        gt_occ = gt != FREE_CLASS
        iou = binary_iou(occ, gt_occ, m).item()
        pred_label = torch.where(occ, sem, torch.full_like(sem, FREE_CLASS))
        miou, _ = semantic_miou(pred_label, gt, m, num_classes)
        out[k] = {"iou": iou, "miou": float(miou)}
        ious.append(iou)
        mious.append(float(miou))
    out["avg"] = {"iou": sum(ious) / len(ious), "miou": sum(mious) / len(mious)}
    return out
