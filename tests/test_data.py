import math

import torch

from rpgwm.data.nuscenes_occ import (SyntheticSequenceDataset, actions_from_poses,
                                     relative_transform)


def pose(x, y, yaw):
    T = torch.eye(4)
    T[0, 0], T[0, 1] = math.cos(yaw), -math.sin(yaw)
    T[1, 0], T[1, 1] = math.sin(yaw), math.cos(yaw)
    T[0, 3], T[1, 3] = x, y
    return T


def test_actions_straight_drive():
    """Ego drives +x at 10 m/s in world frame: dx=5 m per 0.5 s step, dy=dyaw=0."""
    poses = torch.stack([pose(5.0 * t, 0.0, 0.0) for t in range(4)])
    a = actions_from_poses(poses)
    assert a.shape == (3, 4)
    assert torch.allclose(a[:, 0], torch.full((3,), 5.0), atol=1e-5)   # dx
    assert torch.allclose(a[:, 1], torch.zeros(3), atol=1e-5)          # dy
    assert torch.allclose(a[:, 2], torch.zeros(3), atol=1e-5)          # dyaw
    assert torch.allclose(a[:, 3], torch.full((3,), 10.0), atol=1e-4)  # speed


def test_actions_are_ego_frame():
    """Same forward motion but heading 90°: step translation must still be
    'forward' (dx) in the ego frame, not world +y."""
    poses = torch.stack([pose(0.0, 5.0 * t, math.pi / 2) for t in range(3)])
    a = actions_from_poses(poses)
    assert torch.allclose(a[:, 0], torch.full((2,), 5.0), atol=1e-5)
    assert torch.allclose(a[:, 1], torch.zeros(2), atol=1e-5)


def test_relative_transform_roundtrip():
    """A point fixed in the world must land at the right future-ego coords."""
    p_cur, p_fut = pose(0, 0, 0), pose(10, 0, math.pi / 2)
    R, t = relative_transform(p_cur, p_fut)
    world_pt = torch.tensor([12.0, 3.0, 0.0])   # world == current ego frame here
    ego_fut = R @ world_pt + t
    # future ego at (10,0) heading +y: point is 2 ahead in world-x -> ego-fut
    # x_fut = rotate(-90°) applied to offset (2, 3) -> (3, -2)
    assert torch.allclose(ego_fut[:2], torch.tensor([3.0, -2.0]), atol=1e-5)


def test_synthetic_dataset_contract():
    ds = SyntheticSequenceDataset(num_items=3, n_gaussians=16, feat_dim=8,
                                  resolution=(8, 8, 4), future_frames=2)
    item = ds[0]
    V = 8 * 8 * 4
    assert item["state_mu"].shape == (16, 3)
    assert item["actions"].shape == (2, 4)
    assert item["ego_rot"].shape == (2, 3, 3)
    assert item["labels"].shape == (2, V) and item["masks"].shape == (2, V)
    # determinism: same idx -> same item
    again = ds[0]
    assert torch.equal(item["state_mu"], again["state_mu"])
    assert torch.equal(item["labels"], again["labels"])
