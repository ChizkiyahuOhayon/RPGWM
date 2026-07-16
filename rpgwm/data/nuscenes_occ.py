"""Sequence dataset for stage-B rollout training.

Design (remote-GPU discipline): the training loop never touches the nuScenes
devkit. A one-off server script (`scripts/build_index.py`) walks the devkit and
writes a plain JSON index; encoder inference (`scripts/dump_gaussians.py`)
caches one GaussianState .pt per keyframe. This dataset then only reads:
  - the index JSON (scene -> ordered sample tokens + 4x4 ego poses),
  - cached Gaussian states  <gaussian_dir>/<sample_token>.pt,
  - Occ3D labels            <occ3d_root>/gts/<scene>/<token>/labels.npz
    (semantics [200,200,16] uint8 with 17=free, mask_camera same shape).

`SyntheticSequenceDataset` mirrors the exact item contract on random data so
the full training loop is CPU-testable before anything ships to the A40s.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FREE_CLASS = 17


# --------------------------------------------------------------------------
# pose geometry
# --------------------------------------------------------------------------
def relative_transform(pose_cur: torch.Tensor, pose_fut: torch.Tensor):
    """4x4 world-from-ego poses -> (rot [3,3], trans [3]) mapping CURRENT-ego
    coordinates into FUTURE-ego coordinates: x_fut = R x_cur + t."""
    rel = torch.linalg.inv(pose_fut) @ pose_cur
    return rel[:3, :3], rel[:3, 3]


def actions_from_poses(poses: torch.Tensor, dt: float = 0.5) -> torch.Tensor:
    """poses [T, 4, 4] -> actions [T-1, 4] = (dx, dy, dyaw, speed), each step's
    translation expressed in that step's starting ego frame."""
    out = []
    for t in range(poses.shape[0] - 1):
        rel = torch.linalg.inv(poses[t]) @ poses[t + 1]   # ego_t -> ego_{t+1} motion
        dx, dy = rel[0, 3].item(), rel[1, 3].item()
        dyaw = math.atan2(rel[1, 0].item(), rel[0, 0].item())
        speed = math.hypot(dx, dy) / dt
        out.append([dx, dy, dyaw, speed])
    return torch.tensor(out, dtype=torch.float32)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------
class OccSequenceDataset(Dataset):
    """Windows of (H history frames, current frame, K future frames).

    Index JSON: [{"scene": str, "tokens": [...], "poses": [[4x4], ...],
    "cams": [{cam_name: {"img": path, "ego2img": [4x4]}}, ...] (optional)}, ...]
    in temporal order at 2 Hz.

    history_frames > 0 adds the streaming-encoder prerequisites to each item:
      hist_prev2cur [H, 4, 4] — full SE(3) mapping frame (t-H+h)'s ego coords
      into frame (t-H+h+1)'s ego coords, h = 0..H-1 (the last one lands on the
      current frame). The encoder consumes these one by one in warp_slots.
    Camera records (image paths + ego->image matrices) are not collated into
    the tensor item; fetch them with cam_records(idx) in the stage-A loader.
    """

    STATE_KEYS = ("mu", "log_scale", "quat", "opacity", "sem", "feat")

    def __init__(self, index_file: str, gaussian_dir: str, occ3d_root: str,
                 future_frames: int = 6, history_frames: int = 0):
        self.gaussian_dir = Path(gaussian_dir)
        self.occ3d_root = Path(occ3d_root)
        self.future = future_frames
        self.history = history_frames
        scenes = json.loads(Path(index_file).read_text())
        self.windows = []
        for s in scenes:
            n = len(s["tokens"])
            poses = torch.tensor(s["poses"], dtype=torch.float32)
            cams = s.get("cams")
            for i in range(history_frames, n - future_frames):
                lo, hi = i - history_frames, i + future_frames + 1
                self.windows.append((s["scene"], s["tokens"][lo:hi], poses[lo:hi],
                                     cams[lo:hi] if cams else None))

    def __len__(self):
        return len(self.windows)

    def cam_records(self, idx: int):
        """Per-frame camera dicts for the H+1 non-future frames of window
        idx (history..current), or None when the index has no camera info."""
        cams = self.windows[idx][3]
        return cams[:self.history + 1] if cams else None

    def _load_labels(self, scene: str, token: str):
        d = np.load(self.occ3d_root / "gts" / scene / token / "labels.npz")
        return (torch.from_numpy(d["semantics"].astype(np.int64)),
                torch.from_numpy(d["mask_camera"].astype(bool)))

    def __getitem__(self, idx: int):
        scene, tokens, poses, _ = self.windows[idx]
        H = self.history
        state = torch.load(self.gaussian_dir / f"{tokens[H]}.pt", map_location="cpu")
        item = {f"state_{k}": state[k] for k in self.STATE_KEYS}
        item["actions"] = actions_from_poses(poses[H:])                 # [K, 4]
        if H > 0:
            hist = []
            for h in range(H):
                r, t = relative_transform(poses[h], poses[h + 1])
                T = torch.eye(4)
                T[:3, :3], T[:3, 3] = r, t
                hist.append(T)
            item["hist_prev2cur"] = torch.stack(hist)                   # [H, 4, 4]
        rots, transes, labels, masks = [], [], [], []
        for k in range(1, self.future + 1):
            r, t = relative_transform(poses[H], poses[H + k])
            rots.append(r)
            transes.append(t)
            lab, m = self._load_labels(scene, tokens[H + k])
            labels.append(lab.reshape(-1))
            masks.append(m.reshape(-1))
        item["ego_rot"] = torch.stack(rots)                              # [K, 3, 3]
        item["ego_trans"] = torch.stack(transes)                         # [K, 3]
        item["labels"] = torch.stack(labels)                             # [K, V]
        item["masks"] = torch.stack(masks)                               # [K, V]
        return item


# --------------------------------------------------------------------------
# synthetic data (CPU tests / smoke runs)
# --------------------------------------------------------------------------
class SyntheticSequenceDataset(Dataset):
    """Same item contract as OccSequenceDataset, random content. The GT labels
    are derived from the (noisily advected) initial Gaussians so that learning
    signal exists and copy-last-frame is beatable in principle."""

    def __init__(self, num_items: int = 8, n_gaussians: int = 64,
                 num_classes: int = 17, feat_dim: int = 32,
                 resolution=(16, 16, 8), extent: float = 8.0,
                 future_frames: int = 3, history_frames: int = 0, seed: int = 0):
        self.num_items = num_items
        self.n = n_gaussians
        self.c = num_classes
        self.d = feat_dim
        self.res = tuple(resolution)
        self.extent = extent
        self.future = future_frames
        self.history = history_frames
        self.seed = seed

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 10007 + idx)
        V = int(np.prod(self.res))
        from rpgwm.models.gaussians import GaussianState
        s = GaussianState.random(1, self.n, self.c, self.d, extent=self.extent,
                                 generator=g)
        item = {"state_mu": s.mu[0], "state_log_scale": s.log_scale[0],
                "state_quat": s.quat[0], "state_opacity": s.opacity[0],
                "state_sem": s.sem[0], "state_feat": s.feat[0]}
        item["actions"] = torch.randn(self.future, 4, generator=g) * 0.2
        item["ego_rot"] = torch.eye(3).expand(self.future, 3, 3).clone()
        item["ego_trans"] = torch.zeros(self.future, 3)
        if self.history > 0:
            hist = []
            for _ in range(self.history):  # small planar ego motion per step
                yaw = (torch.rand(1, generator=g).item() - 0.5) * 0.2
                T = torch.eye(4)
                c, s = math.cos(yaw), math.sin(yaw)
                T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
                T[0, 3] = (torch.rand(1, generator=g).item() - 0.2) * 2.0
                T[1, 3] = (torch.rand(1, generator=g).item() - 0.5) * 0.5
                hist.append(T)
            item["hist_prev2cur"] = torch.stack(hist)
        labels = torch.full((self.future, V), FREE_CLASS, dtype=torch.long)
        occupied = torch.rand(self.future, V, generator=g) < 0.15
        labels[occupied] = torch.randint(0, self.c, (int(occupied.sum()),), generator=g)
        item["labels"] = labels
        item["masks"] = torch.rand(self.future, V, generator=g) < 0.9
        return item
