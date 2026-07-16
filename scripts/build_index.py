#!/usr/bin/env python3
"""One-off (server): build the plain-JSON sequence index from the nuScenes
devkit so training never imports the devkit.

Usage:
  python scripts/build_index.py --nuscenes-root data/nuscenes \
      --version v1.0-mini --split train --out data/index_mini_train.json

Output: [{"scene": <scene name>, "tokens": [sample_token, ...],
          "poses": [[4x4 ego pose], ...],
          "cams": [{CAM_NAME: {"img": <relpath>, "ego2img": [4x4]}}, ...]}, ...]
(keyframes, 2 Hz, temporal order)
Poses are LIDAR_TOP ego poses (world-from-ego), the convention Occ3D uses.
ego2img maps points in the LIDAR-timestamp ego frame to image pixels:
  K_pad @ inv(sensor2ego_cam) @ inv(egopose_cam) @ egopose_lidar
(accounts for the small ego motion between the lidar and camera triggers).
Pass --no-cams to skip camera records (stage-B-only indexes).
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
    ap.add_argument("--no-cams", action="store_true",
                    help="skip camera records (smaller index, stage B only)")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion

    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=True)
    split_key = {"v1.0-mini": {"train": "mini_train", "val": "mini_val"},
                 "v1.0-trainval": {"train": "train", "val": "val"}}[args.version][args.split]
    wanted = set(create_splits_scenes()[split_key])

    CAM_NAMES = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                 "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

    def pose_matrix(rec):
        T = np.eye(4)
        T[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
        T[:3, 3] = rec["translation"]
        return T

    def cam_record(sample, egopose_lidar):
        rec = {}
        for name in CAM_NAMES:
            sd = nusc.get("sample_data", sample["data"][name])
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            egopose_cam = pose_matrix(nusc.get("ego_pose", sd["ego_pose_token"]))
            K = np.eye(4)
            K[:3, :3] = np.array(cs["camera_intrinsic"])
            ego2img = (K @ np.linalg.inv(pose_matrix(cs))
                       @ np.linalg.inv(egopose_cam) @ egopose_lidar)
            rec[name] = {"img": sd["filename"], "ego2img": ego2img.tolist()}
        return rec

    out = []
    for scene in nusc.scene:
        if scene["name"] not in wanted:
            continue
        tokens, poses, cams = [], [], []
        tok = scene["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            T = pose_matrix(nusc.get("ego_pose", sd["ego_pose_token"]))
            tokens.append(tok)
            poses.append(T.tolist())
            if not args.no_cams:
                cams.append(cam_record(sample, T))
            tok = sample["next"]
        entry = {"scene": scene["name"], "tokens": tokens, "poses": poses}
        if not args.no_cams:
            entry["cams"] = cams
        out.append(entry)

    with open(args.out, "w") as f:
        json.dump(out, f)
    n_frames = sum(len(s["tokens"]) for s in out)
    print(f"wrote {args.out}: {len(out)} scenes, {n_frames} keyframes")


if __name__ == "__main__":
    main()
