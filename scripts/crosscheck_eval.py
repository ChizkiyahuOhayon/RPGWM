#!/usr/bin/env python3
"""Cross-check our forecasting protocol (rpgwm/eval/forecast.py) against an
official OccWorld-lineage eval on the SAME predictions (SERVER_RUNBOOK step 3).

Expected dump layout (one npz per sample, produced by a thin hook added to the
official repo's eval loop — their eval already holds these arrays in memory):
  <dump_dir>/<sample_token>.npz with
    pred [K, 200, 200, 16] uint8 semantic labels, 17 = free
    gt   [K, 200, 200, 16] uint8
    mask [K, 200, 200, 16] uint8 camera-visibility

Usage:
  python scripts/crosscheck_eval.py --dump-dir out/occworld_pred \
      [--official-miou 17.14]     # their reported number for the same set
Acceptance: |ours - official| < 0.05 per horizon.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpgwm.eval.forecast import FREE_CLASS, forecast_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--official-miou", type=float, default=None)
    args = ap.parse_args()

    files = sorted(Path(args.dump_dir).glob("*.npz"))
    assert files, f"no npz files in {args.dump_dir}"
    per_step_acc: dict[int, list[dict]] = {}
    for f in files:
        d = np.load(f)
        pred = torch.from_numpy(d["pred"].astype(np.int64))
        gt = torch.from_numpy(d["gt"].astype(np.int64))
        mask = torch.from_numpy(d["mask"].astype(bool))
        K = pred.shape[0]
        frames = [((pred[k] != FREE_CLASS).reshape(-1), pred[k].reshape(-1)) for k in range(K)]
        out = forecast_scores(frames, [gt[k].reshape(-1) for k in range(K)],
                              [mask[k].reshape(-1) for k in range(K)])
        for k in range(K):
            per_step_acc.setdefault(k, []).append(out[k])

    print(f"{len(files)} samples")
    for k, rows in sorted(per_step_acc.items()):
        iou = sum(r["iou"] for r in rows) / len(rows)
        miou = sum(r["miou"] for r in rows) / len(rows)
        print(f"  step {k} ({(k+1)*0.5:.1f}s): IoU {iou*100:.2f}  mIoU {miou*100:.2f}")
    all_miou = [r["miou"] for rows in per_step_acc.values() for r in rows]
    ours = 100 * sum(all_miou) / len(all_miou)
    print(f"  avg mIoU {ours:.2f}")
    if args.official_miou is not None:
        diff = abs(ours - args.official_miou)
        print(f"  official {args.official_miou:.2f}  |diff| {diff:.3f} "
              f"-> {'PASS' if diff < 0.05 else 'FAIL (investigate before reporting Table-1 numbers)'}")


if __name__ == "__main__":
    main()
