"""Lovász loss unit tests (debt item 1: replaced the soft-IoU stand-in —
must be green before stage-A training starts)."""
import torch
import torch.nn.functional as F

from rpgwm.losses import lovasz_binary, lovasz_softmax, occupancy_recon_loss


def test_binary_perfect_is_zero_and_worse_is_larger():
    gt = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    perfect = lovasz_binary(gt.clone(), gt)
    assert perfect.abs() < 1e-6
    bad = lovasz_binary(1.0 - gt, gt)          # maximally wrong
    mid = lovasz_binary(torch.full_like(gt, 0.5), gt)
    assert bad > mid > perfect


def test_binary_empty_mask_is_zero():
    z = torch.zeros(0)
    assert lovasz_binary(z, z) == 0


def test_softmax_perfect_is_zero_and_absent_classes_ignored():
    labels = torch.tensor([0, 0, 2, 2])        # class 1 absent
    probs = F.one_hot(labels, 4).float()
    assert lovasz_softmax(probs, labels).abs() < 1e-6
    wrong = torch.roll(probs, 1, dims=-1)
    assert lovasz_softmax(wrong, labels) > 0.5


def test_gradients_flow_through_both_terms():
    torch.manual_seed(0)
    B, V, C = 1, 64, 5
    occ_logit = torch.randn(B, V, requires_grad=True)
    sem_logit = torch.randn(B, V, C + 1, requires_grad=True)
    gt = torch.randint(0, C + 1, (B, V))       # C = free class here
    visible = torch.rand(B, V) < 0.8
    loss = occupancy_recon_loss(torch.sigmoid(occ_logit), sem_logit, gt,
                                visible, free_class=C)
    loss.backward()
    assert occ_logit.grad is not None and occ_logit.grad.abs().sum() > 0
    assert sem_logit.grad is not None and sem_logit.grad.abs().sum() > 0


def test_recon_loss_ranks_better_predictions_lower():
    torch.manual_seed(1)
    B, V, C = 1, 128, 5
    gt = torch.randint(0, C + 1, (B, V))
    visible = torch.ones(B, V, dtype=torch.bool)
    gt_occ = (gt != C).float()
    sem_perfect = F.one_hot(gt.clamp_max(C - 1), C).float() * 8.0
    good = occupancy_recon_loss(gt_occ * 0.98 + 0.01, sem_perfect, gt,
                                visible, free_class=C)
    noisy = occupancy_recon_loss((gt_occ * 0.6 + 0.2), sem_perfect * 0.1, gt,
                                 visible, free_class=C)
    assert good < noisy
