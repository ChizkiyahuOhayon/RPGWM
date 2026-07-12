import torch

from rpgwm.eval.forecast import FREE_CLASS, binary_iou, forecast_scores, semantic_miou
from rpgwm.losses import occupancy_recon_loss, plan_sufficiency_loss, rho_regression_loss


def test_binary_iou_perfect_and_empty():
    gt = torch.tensor([True, True, False, False])
    mask = torch.ones(4, dtype=torch.bool)
    assert binary_iou(gt, gt, mask) == 1.0
    assert binary_iou(torch.zeros(4, dtype=torch.bool), gt, mask) == 0.0


def test_iou_respects_visibility_mask():
    gt = torch.tensor([True, True, False, False])
    pred = torch.tensor([True, False, False, False])  # misses gt[1]
    mask = torch.tensor([True, False, True, True])    # ...but gt[1] is invisible
    assert binary_iou(pred, gt, mask) == 1.0


def test_semantic_miou_ignores_absent_classes():
    gt = torch.tensor([0, 0, 1, FREE_CLASS])
    pred = torch.tensor([0, 0, 1, FREE_CLASS])
    miou, per_class = semantic_miou(pred, gt, torch.ones(4, dtype=torch.bool))
    assert float(miou) == 1.0
    assert torch.isnan(per_class[5])  # class 5 absent everywhere -> excluded


def test_forecast_scores_structure():
    V = 64
    gt = torch.randint(0, 3, (V,))
    frames = [((gt != FREE_CLASS), gt.clone()) for _ in range(3)]
    masks = [torch.ones(V, dtype=torch.bool)] * 3
    out = forecast_scores(frames, [gt] * 3, masks)
    assert set(out.keys()) == {0, 1, 2, "avg"}
    assert out["avg"]["iou"] == 1.0 and out["avg"]["miou"] == 1.0


def test_recon_loss_decreases_for_better_prediction():
    torch.manual_seed(0)
    B, V, C = 1, 200, 17
    gt = torch.randint(0, C + 1, (B, V))
    visible = torch.ones(B, V, dtype=torch.bool)
    gt_occ = (gt != FREE_CLASS).float()
    sem_perfect = torch.nn.functional.one_hot(gt.clamp_max(C - 1), C).float() * 10
    good = occupancy_recon_loss(gt_occ * 0.95 + 0.02, sem_perfect, gt, visible)
    bad = occupancy_recon_loss((1 - gt_occ) * 0.95 + 0.02, sem_perfect, gt, visible)
    assert good < bad


def test_rho_loss_only_counts_valid_slots():
    rho = torch.tensor([[0.2, 0.9]])
    target = torch.tensor([[0.2, 0.0]])
    valid = torch.tensor([[True, False]])
    assert rho_regression_loss(rho, target, valid) == 0.0


def test_plan_sufficiency_zero_when_identical():
    traj = torch.randn(2, 6, 2)
    logits = torch.randn(2, 20)
    loss = plan_sufficiency_loss(traj, traj.detach(), logits, logits.detach())
    assert loss.abs() < 1e-6
