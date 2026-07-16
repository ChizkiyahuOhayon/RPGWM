"""Stage-A image datasets: multi-camera frames + streaming history + Occ3D
labels for the current frame.

Image pipeline = the BEVDet/S2GO standard (NOT naive resize — 900x1600 to
256x704 directly would squash the aspect ratio by ~35% and invalidate every
ego2img matrix): scale by final_W / raw_W (704/1600 = 0.44 -> 396x704), then
crop `crop_top` rows from the top (396-256 = 140, sky region). The same
scale/crop is applied analytically to the ego2img matrices:
  u' = s*u          -> row0 *= s
  v' = s*v - crop   -> row1 = s*row1 - crop*row2
Normalization = GF-2 / torchvision ImageNet (config/_base_/surroundocc.py:8:
mean [123.675,116.28,103.53], std [58.395,57.12,57.375], to_rgb) applied
after /255. Every number is config-driven; nothing here is hardcoded policy.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .nuscenes_occ import FREE_CLASS, relative_transform

CAM_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # = GF-2's [123.675,116.28,103.53]/255
IMAGENET_STD = (0.229, 0.224, 0.225)    # = GF-2's [58.395,57.12,57.375]/255


@dataclass
class ImagePipeline:
    """Deterministic val-style pipeline (train-time jitter is a later,
    config-gated addition — encoder stage is no-innovation)."""
    raw_hw: tuple[int, int] = (900, 1600)
    final_hw: tuple[int, int] = (256, 704)

    @property
    def scale(self) -> float:
        return self.final_hw[1] / self.raw_hw[1]

    @property
    def crop_top(self) -> int:
        resized_h = int(round(self.raw_hw[0] * self.scale))
        crop = resized_h - self.final_hw[0]
        if crop < 0:
            raise ValueError(f"final height {self.final_hw[0]} exceeds resized "
                             f"height {resized_h} — bad pipeline config")
        return crop

    def apply_to_image(self, img: "np.ndarray") -> torch.Tensor:
        """HWC uint8 RGB (raw_hw) -> normalized CHW float in final_hw."""
        from PIL import Image
        h, w = self.final_hw[0], self.final_hw[1]
        resized_h = int(round(img.shape[0] * self.scale))
        pil = Image.fromarray(img).resize((w, resized_h), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32)[self.crop_top:self.crop_top + h] / 255.0
        arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        return torch.from_numpy(arr).permute(2, 0, 1).float()

    def apply_to_projection(self, ego2img: torch.Tensor) -> torch.Tensor:
        """ego2img [..., 4, 4] in raw pixels -> final (scaled+cropped) pixels."""
        out = ego2img.clone()
        out[..., 0, :] = ego2img[..., 0, :] * self.scale
        out[..., 1, :] = ego2img[..., 1, :] * self.scale \
            - self.crop_top * ego2img[..., 2, :]
        return out

    @property
    def image_wh(self) -> torch.Tensor:
        return torch.tensor([float(self.final_hw[1]), float(self.final_hw[0])])


class NuScenesImageSequenceDataset(Dataset):
    """Item contract (stage A, streaming):
      images      [H+1, 6, 3, h, w]   history..current, pipeline applied
      projection  [H+1, 6, 4, 4]      ego2img adjusted to pipeline pixels
      hist_prev2cur [H, 4, 4]         frame h ego -> frame h+1 ego
      label / mask  [V] long / bool   Occ3D of the CURRENT frame
    Index JSON must carry "cams" records (scripts/build_index.py without
    --no-cams)."""

    def __init__(self, index_file: str, nuscenes_root: str, occ3d_root: str,
                 history_frames: int = 0, pipeline: ImagePipeline | None = None,
                 cam_order=CAM_ORDER):
        self.root = Path(nuscenes_root)
        self.occ3d_root = Path(occ3d_root)
        self.history = history_frames
        self.pipe = pipeline or ImagePipeline()
        self.cam_order = cam_order
        scenes = json.loads(Path(index_file).read_text())
        self.windows = []
        for s in scenes:
            if "cams" not in s:
                raise ValueError(f"index has no cams for scene {s['scene']} — "
                                 f"rebuild with scripts/build_index.py (no --no-cams)")
            poses = torch.tensor(s["poses"], dtype=torch.float32)
            for i in range(history_frames, len(s["tokens"])):
                lo = i - history_frames
                self.windows.append((s["scene"], s["tokens"][lo:i + 1],
                                     poses[lo:i + 1], s["cams"][lo:i + 1]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        from PIL import Image
        scene, tokens, poses, cams = self.windows[idx]
        frames, projs = [], []
        for rec in cams:
            imgs, mats = [], []
            for cam in self.cam_order:
                img = np.asarray(Image.open(self.root / rec[cam]["img"]).convert("RGB"))
                imgs.append(self.pipe.apply_to_image(img))
                mats.append(self.pipe.apply_to_projection(
                    torch.tensor(rec[cam]["ego2img"], dtype=torch.float32)))
            frames.append(torch.stack(imgs))
            projs.append(torch.stack(mats))
        item = {"images": torch.stack(frames), "projection": torch.stack(projs),
                "image_wh": self.pipe.image_wh}
        if self.history > 0:
            hist = []
            for h in range(self.history):
                r, t = relative_transform(poses[h], poses[h + 1])
                T = torch.eye(4)
                T[:3, :3], T[:3, 3] = r, t
                hist.append(T)
            item["hist_prev2cur"] = torch.stack(hist)
        d = np.load(self.occ3d_root / "gts" / scene / tokens[-1] / "labels.npz")
        item["label"] = torch.from_numpy(d["semantics"].astype(np.int64)).reshape(-1)
        item["mask"] = torch.from_numpy(d["mask_camera"].astype(bool)).reshape(-1)
        return item


class SyntheticImageSequenceDataset(Dataset):
    """Same item contract on random data — the stage-A trainer must run on
    CPU before it ships to the A40s."""

    def __init__(self, num_items: int = 4, num_cams: int = 2, image_hw=(64, 128),
                 history_frames: int = 1, resolution=(16, 16, 8),
                 num_classes: int = 17, seed: int = 0):
        self.num_items = num_items
        self.num_cams = num_cams
        self.hw = tuple(image_hw)
        self.history = history_frames
        self.res = tuple(resolution)
        self.c = num_classes
        self.seed = seed

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 7919 + idx)
        F_, V = self.history + 1, int(np.prod(self.res))
        h, w = self.hw
        K = torch.eye(4)
        K[0, 0] = K[1, 1] = w / 2.0
        K[0, 2], K[1, 2] = w / 2.0, h / 2.0
        R = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        E = torch.eye(4)
        E[:3, :3] = R
        proj = (K @ E).expand(F_, self.num_cams, 4, 4).clone()
        item = {"images": torch.rand(F_, self.num_cams, 3, h, w, generator=g),
                "projection": proj,
                "image_wh": torch.tensor([float(w), float(h)])}
        if self.history > 0:
            hist = []
            for _ in range(self.history):
                yaw = (torch.rand(1, generator=g).item() - 0.5) * 0.1
                T = torch.eye(4)
                c, s = math.cos(yaw), math.sin(yaw)
                T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
                T[0, 3] = torch.rand(1, generator=g).item()
                hist.append(T)
            item["hist_prev2cur"] = torch.stack(hist)
        label = torch.full((V,), FREE_CLASS, dtype=torch.long)
        occ = torch.rand(V, generator=g) < 0.2
        label[occ] = torch.randint(0, self.c, (int(occ.sum()),), generator=g)
        item["label"] = label
        item["mask"] = torch.rand(V, generator=g) < 0.9
        return item
