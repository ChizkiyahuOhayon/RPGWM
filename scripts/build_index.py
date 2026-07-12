#!/usr/bin/env python3
"""One-off (server): build the plain-JSON sequence index from the nuScenes
devkit so training never imports the devkit.

Usage:
  python scripts/build_index.py --nuscenes-root data/nuscenes \
      --version v1.0-mini --split train --out data/index_mini_train.json

Output: [{"scene": <scene name>, "tokens": [sample_token, ...],
          "poses": [[4x4 ego pose], ...]}, ...]  (keyframes, 2 Hz, temporal order)
Poses are LIDAR_TOP ego poses (world-from-ego), the convention Occ3D uses.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion

    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=True)
    split_key = {"v1.0-mini": {"train": "mini_train", "val": "mini_val"},
                 "v1.0-trainval": {"train": "train", "val": "val"}}[args.version][args.split]
    wanted = set(create_splits_scenes()[split_key])

    out = []
    for scene in nusc.scene:
        if scene["name"] not in wanted:
            continue
        tokens, poses = [], []
        tok = scene["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            T = np.eye(4)
            T[:3, :3] = Quaternion(ep["rotation"]).rotation_matrix
            T[:3, 3] = ep["translation"]
            tokens.append(tok)
            poses.append(T.tolist())
            tok = sample["next"]
        out.append({"scene": scene["name"], "tokens": tokens, "poses": poses})

    with open(args.out, "w") as f:
        json.dump(out, f)
    n_frames = sum(len(s["tokens"]) for s in out)
    print(f"wrote {args.out}: {len(out)} scenes, {n_frames} keyframes")


if __name__ == "__main__":
    main()
