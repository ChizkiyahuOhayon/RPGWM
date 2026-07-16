#!/usr/bin/env python3
"""A-G stage probe (W1 hard deliverable, runs on the SECOND A40 in parallel
with stage A): single-clip Vista teacher-feature extraction smoke + throughput
measurement, to lock the A-G resolution/scope before the full 28k-clip cache.

What it does:
  1. builds frozen Vista from third_party/Vista (their sample_utils.init_model)
  2. registers forward hooks on the requested U-Net output blocks
  3. runs ONE single-forward denoise at early step t* on a dummy (or real)
     conditioning clip
  4. reports: per-block feature shapes, wall time per clip, bytes per clip at
     fp16, projected cache size for --num-clips
Fail fast: every failure exits non-zero within seconds with the cause — env
mismatch, missing weights, or hook-name drift against the Vista version.

STATUS: written against Vista @ third_party/Vista (cc9821b); NOT runnable on
the CPU dev box (needs Vista env + weights). First server run may need the
--list-blocks pass to pin exact hook names. That is expected W1 work, not a
silent assumption.

Usage (server, vista env):
  python scripts/ag_probe_vista.py --vista-root third_party/Vista \
      --ckpt ckpts/vista.safetensors --t-star 1 --list-blocks
  python scripts/ag_probe_vista.py --vista-root third_party/Vista \
      --ckpt ckpts/vista.safetensors --t-star 1 \
      --blocks output_blocks.6 output_blocks.9 --num-clips 28130
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def die(msg: str, code: int = 1):
    print(f"[ag_probe] FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vista-root", default="third_party/Vista")
    ap.add_argument("--ckpt", required=True, help="vista.safetensors path")
    ap.add_argument("--config", default="configs/inference/vista.yaml")
    ap.add_argument("--t-star", type=int, default=1, choices=[1, 5],
                    help="early denoise step (DriveLaW Tab.6 grid)")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--n-frames", type=int, default=7,
                    help="1 current + K=6 future @2Hz teacher targets")
    ap.add_argument("--blocks", nargs="*", default=[],
                    help="U-Net submodule names to hook (see --list-blocks)")
    ap.add_argument("--list-blocks", action="store_true",
                    help="print U-Net block names + shapes, then exit")
    ap.add_argument("--num-clips", type=int, default=28130,
                    help="for cache-size projection (nuScenes train keyframes)")
    ap.add_argument("--out", default="outputs/ag_probe/report.json")
    args = ap.parse_args()

    root = Path(args.vista_root).resolve()
    if not (root / "vwm").is_dir():
        die(f"{root} has no vwm/ — wrong --vista-root?")
    if not Path(args.ckpt).exists():
        die(f"checkpoint not found: {args.ckpt} (W1: download to ckpts/ first)")
    sys.path.insert(0, str(root))

    t0 = time.time()
    try:
        from sample_utils import init_model  # Vista's own loader
    except Exception as e:  # noqa: BLE001
        die(f"cannot import Vista sample_utils ({e}) — wrong env? "
            f"(needs the vista conda env, not rpgwm/mm-stack)")

    version_dict = {"config": str(root / args.config), "ckpt": args.ckpt,
                    "options": {}}
    try:
        model = init_model(version_dict)
    except Exception as e:  # noqa: BLE001
        die(f"Vista init_model failed: {e}")
    model.eval().requires_grad_(False)
    unet = model.model.diffusion_model  # SVD-family UNet
    print(f"[ag_probe] model up in {time.time()-t0:.1f}s")

    if args.list_blocks:
        for name, mod in unet.named_children():
            print(f"  unet.{name}: {type(mod).__name__}")
        for name, _ in unet.named_modules():
            if name.count(".") <= 1 and ("output_blocks" in name or "middle_block" in name):
                print(f"  hookable: {name}")
        return

    if not args.blocks:
        die("no --blocks given (run --list-blocks first to pin names)")

    feats: dict[str, torch.Tensor] = {}
    hooks = []
    mods = dict(unet.named_modules())
    for b in args.blocks:
        if b not in mods:
            die(f"block '{b}' not in this Vista build; candidates: "
                f"{[n for n in mods if 'output_blocks' in n and n.count('.') <= 1][:8]}")
        hooks.append(mods[b].register_forward_hook(
            lambda m, i, o, name=b: feats.__setitem__(
                name, (o[0] if isinstance(o, tuple) else o).detach())))

    device = "cuda"
    model = model.to(device)
    # dummy conditioning clip at the probe resolution; the full extractor
    # feeds real nuScenes front-cam frames + GT ego actions through Vista's
    # conditioning pipeline — the probe only measures shapes and throughput.
    frames = torch.randn(args.n_frames, 3, args.height, args.width, device=device)

    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        z = model.encode_first_stage(frames)
        sigma = model.denoiser.idx_to_sigma(torch.tensor([args.t_star], device=device)) \
            if hasattr(model.denoiser, "idx_to_sigma") else None
        noised = z + torch.randn_like(z) * (sigma if sigma is not None else 0.1)
        cond = {"crossattn": torch.zeros(args.n_frames, 1, 1024, device=device),
                "vector": torch.zeros(args.n_frames, 256, device=device),
                "concat": torch.zeros_like(z)}
        try:
            _ = unet(noised, timesteps=torch.full((args.n_frames,), float(args.t_star),
                                                  device=device), context=cond["crossattn"])
        except TypeError as e:
            die(f"UNet signature drift vs this probe ({e}); adapt the call in "
                f"ag_probe_vista.py:{sys._getframe().f_lineno} to this Vista "
                f"version — that is the point of the probe")
    torch.cuda.synchronize()
    wall = time.time() - t1

    for h in hooks:
        h.remove()
    if not feats:
        die("hooks fired on no block — wrong names?")

    bytes_fp16 = sum(v.numel() * 2 for v in feats.values())
    report = {
        "t_star": args.t_star, "resolution": [args.height, args.width],
        "n_frames": args.n_frames,
        "block_shapes": {k: list(v.shape) for k, v in feats.items()},
        "seconds_per_clip": round(wall, 3),
        "mb_per_clip_fp16": round(bytes_fp16 / 2**20, 2),
        "projected_cache_gb": round(bytes_fp16 * args.num_clips / 2**30, 1),
        "projected_extract_days_1gpu": round(wall * args.num_clips / 86400, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    # A-G budget guards (§3.1: 3-6 MB/clip, 90-170 GB total, 12 GPU-days incl.
    # DD-2 quarter split)
    if report["projected_cache_gb"] > 200:
        print("[ag_probe] WARNING: cache projection exceeds §3.1 budget — "
              "reduce resolution/blocks before the full run")


if __name__ == "__main__":
    main()
