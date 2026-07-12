import torch
import torch.nn.functional as F

from rpgwm.models.gaussians import GaussianState
from rpgwm.models.splat import VoxelGrid, splat_occupancy, transform_to_future_ego


def single_gaussian_state(mu=(0.0, 0.0, 0.0), log_scale=0.0, opacity=0.9, C=17, D=8):
    """One Gaussian at mu with std = exp(log_scale) meters per axis."""
    return GaussianState(
        mu=torch.tensor([[list(mu)]], dtype=torch.float32),
        log_scale=torch.full((1, 1, 3), float(log_scale)),
        quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        opacity=torch.tensor([[opacity]]),
        sem=F.one_hot(torch.tensor([[2]]), C).float() * 5.0,
        feat=torch.zeros(1, 1, D),
    )


def small_grid():
    return VoxelGrid(xyz_min=(-4, -4, -4), xyz_max=(4, 4, 4), resolution=(16, 16, 16))


def test_high_prob_at_center_low_far_away():
    s, grid = single_gaussian_state(), small_grid()
    occ_prob, occ_hard, sem = splat_occupancy(s, grid)
    centers = grid.centers()
    near = centers.norm(dim=-1).argmin()
    far = centers.norm(dim=-1).argmax()
    assert occ_prob[0, near] > 0.5
    assert occ_prob[0, far] < 1e-3
    assert occ_hard[0, near] and not occ_hard[0, far]
    # semantics at the occupied voxel vote for the Gaussian's class
    assert sem[0, near].argmax() == 2


def test_dead_slot_contributes_nothing():
    s = single_gaussian_state(opacity=0.01)  # below the 0.05 alive gate
    occ_prob, occ_hard, _ = splat_occupancy(s, small_grid())
    assert occ_prob.max() < 1e-6 and not occ_hard.any()


def test_gradients_flow_to_gaussian_parameters():
    s, grid = single_gaussian_state(), small_grid()
    s.mu.requires_grad_(True)
    s.log_scale.requires_grad_(True)
    occ_prob, _, _ = splat_occupancy(s, grid)
    occ_prob.sum().backward()
    assert s.mu.grad is not None and s.mu.grad.abs().sum() > 0
    assert s.log_scale.grad is not None and s.log_scale.grad.abs().sum() > 0


def test_three_sigma_cutoff():
    """A Gaussian with tiny std must not touch voxels beyond 3 sigma."""
    s, grid = single_gaussian_state(log_scale=-2.0), small_grid()  # std≈0.135 m
    occ_prob, _, _ = splat_occupancy(s, grid)
    centers = grid.centers()
    outside = centers.norm(dim=-1) > 3.0 * 0.14
    assert occ_prob[0, outside].max() < 1e-6


def test_future_ego_transform_pure_translation():
    s = single_gaussian_state(mu=(1.0, 2.0, 0.0))
    rot = torch.eye(3).unsqueeze(0)
    trans = torch.tensor([[10.0, 0.0, 0.0]])
    out = transform_to_future_ego(s, rot, trans)
    assert torch.allclose(out.mu, torch.tensor([[[11.0, 2.0, 0.0]]]))


def test_future_ego_transform_rotation_preserves_cov_pd():
    s = single_gaussian_state()
    yaw = torch.tensor(0.7)
    rot = torch.tensor([[[torch.cos(yaw), -torch.sin(yaw), 0.0],
                         [torch.sin(yaw), torch.cos(yaw), 0.0],
                         [0.0, 0.0, 1.0]]])
    out = transform_to_future_ego(s, rot, torch.zeros(1, 3))
    cov = out.covariance()[0, 0]
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()
