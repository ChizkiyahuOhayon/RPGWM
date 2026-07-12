import torch
import torch.nn.functional as F

from rpgwm.models.error_attribution import (fit_single_step_noise,
                                            per_gaussian_realized_error)
from rpgwm.models.gaussians import GaussianState
from rpgwm.models.splat import VoxelGrid, splat_occupancy

FREE = 17


def two_gaussian_state(C=17, D=8):
    """Gaussian 0 at origin (class 2), Gaussian 1 at (5,5,0) (class 4)."""
    mu = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]]])
    log_scale = torch.zeros(1, 2, 3)          # std = 1 m
    quat = torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]])
    opacity = torch.tensor([[0.9, 0.9]])
    sem = torch.stack([F.one_hot(torch.tensor(2), C), F.one_hot(torch.tensor(4), C)]
                      ).float().unsqueeze(0) * 8.0
    feat = torch.zeros(1, 2, D)
    return GaussianState(mu, log_scale, quat, opacity, sem, feat)


def grid_and_centers():
    grid = VoxelGrid(xyz_min=(-8, -8, -4), xyz_max=(8, 8, 4), resolution=(16, 16, 8))
    return grid, grid.centers()


def gt_from_state(state, grid, cls_map):
    """Build a GT label field that exactly matches the splatted prediction,
    with per-Gaussian classes taken from cls_map."""
    _, occ_hard, sem = splat_occupancy(state, grid)
    gt = torch.full_like(occ_hard[0], FREE, dtype=torch.long)
    lab = sem[0].argmax(-1)
    gt[occ_hard[0]] = torch.tensor(cls_map)[lab[occ_hard[0]].clamp_max(len(cls_map) - 1)]
    return gt.unsqueeze(0)


def test_correct_prediction_has_low_error():
    state = two_gaussian_state()
    grid, centers = grid_and_centers()
    _, occ_hard, _ = splat_occupancy(state, grid)
    ident = list(range(17))
    gt = gt_from_state(state, grid, ident)             # GT == prediction
    vis = torch.ones_like(gt, dtype=torch.bool)
    out = per_gaussian_realized_error(state, occ_hard, gt, vis, centers)
    assert out["valid"].all()
    assert (out["local_iou"] > 0.99).all()
    assert (out["e"] < 0.2).all()


def test_hallucinated_gaussian_has_high_error():
    """GT says empty everywhere -> a confidently splatted Gaussian is wrong."""
    state = two_gaussian_state()
    grid, centers = grid_and_centers()
    _, occ_hard, _ = splat_occupancy(state, grid)
    gt = torch.full((1, occ_hard.shape[1]), FREE, dtype=torch.long)
    vis = torch.ones_like(gt, dtype=torch.bool)
    out = per_gaussian_realized_error(state, occ_hard, gt, vis, centers)
    assert (out["local_iou"] < 0.01).all()             # predicted occ, gt empty
    assert (out["e"] > 0.9).all()


def test_wrong_semantics_raises_sem_err_only():
    state = two_gaussian_state()
    grid, centers = grid_and_centers()
    _, occ_hard, _ = splat_occupancy(state, grid)
    swapped = list(range(17))
    swapped[2], swapped[4] = 9, 11                     # geometry right, class wrong
    gt = gt_from_state(state, grid, swapped)
    vis = torch.ones_like(gt, dtype=torch.bool)
    out = per_gaussian_realized_error(state, occ_hard, gt, vis, centers)
    assert (out["local_iou"] > 0.99).all()
    assert (out["sem_err"] > 0.9).all()


def test_invisible_support_excluded_from_calibration():
    state = two_gaussian_state()
    grid, centers = grid_and_centers()
    _, occ_hard, _ = splat_occupancy(state, grid)
    gt = gt_from_state(state, grid, list(range(17)))
    vis = torch.zeros_like(gt, dtype=torch.bool)       # nothing visible
    out = per_gaussian_realized_error(state, occ_hard, gt, vis, centers)
    assert not out["has_gt_support"].any()
    assert not out["valid"].any()


def test_center_offset_measures_displacement():
    """Shift GT occupancy by moving the state used to build GT: the measured
    centroid offset must point from prediction toward the GT mass."""
    pred = two_gaussian_state()
    grid, centers = grid_and_centers()
    shifted = two_gaussian_state()
    shifted.mu = shifted.mu + torch.tensor([1.5, 0.0, 0.0])
    gt = gt_from_state(shifted, grid, list(range(17)))
    _, occ_hard, _ = splat_occupancy(pred, grid)
    vis = torch.ones_like(gt, dtype=torch.bool)
    out = per_gaussian_realized_error(pred, occ_hard, gt, vis, centers)
    ok = out["has_gt_support"][0] & ~torch.isnan(out["center_offset"][0]).any(-1)
    assert ok.any()
    assert (out["center_offset"][0][ok][:, 0] > 0.3).all()   # offset toward +x


def test_fit_single_step_noise():
    torch.manual_seed(0)
    offs = torch.randn(5000, 3) * torch.tensor([0.2, 0.3, 0.1])
    var = fit_single_step_noise(offs)
    assert torch.allclose(var, torch.tensor([0.04, 0.09, 0.01]), rtol=0.15)
    # degenerate: too few samples -> prior
    assert torch.allclose(fit_single_step_noise(torch.full((3, 3), float("nan"))),
                          torch.full((3,), 0.04))
