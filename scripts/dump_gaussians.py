#!/usr/bin/env python3
"""Stage A' (server): run the (warm-started) Gaussian encoder over every
keyframe and cache one GaussianState .pt per sample token — the frozen-encoder
feature cache that stage B trains on.

Two modes:
  --encoder gf2     load GaussianFormer-2 from third_party (fill in the two
                    TODO hooks below after SERVER_RUNBOOK step 2 pins the
                    checkpoint + config paths)
  --encoder random  deterministic random states — pipeline smoke test only,
                    lets stage-B plumbing run end-to-end before the encoder
                    integration lands. NEVER use for reported numbers.

Usage:
  python scripts/dump_gaussians.py --index data/index_mini_train.json \
      --out data/gaussian_cache --encoder random --n 6400
"""
import argparse
import json
from pathlib import Path

import torch


def encode_random(token: str, n: int, num_classes: int, feat_dim: int):
    g = torch.Generator().manual_seed(abs(hash(token)) % (2 ** 31))
    from rpgwm.models.gaussians import GaussianState
    s = GaussianState.random(1, n, num_classes, feat_dim, extent=40.0, generator=g)
    return {k: getattr(s, k)[0].contiguous() for k in GaussianState.FIELDS}


def build_gf2_encoder(ckpt: str, config: str):
    """TODO(server, after RUNBOOK step 2): load GaussianFormer-2.
    Expected: sys.path.insert third_party/GaussianFormer, build model from its
    py-config, load ckpt, return a callable(sample_token) -> state dict with
    keys mu/log_scale/quat/opacity/sem/feat mapped from GF-2's
    (mean, scale, rot, opacity, semantics, query_feature)."""
    raise NotImplementedError("wire GaussianFormer-2 here (SERVER_RUNBOOK step 2)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=["gf2", "random"], default="gf2")
    ap.add_argument("--gf2-ckpt", default="ckpts/gaussianformer2.pth")
    ap.add_argument("--gf2-config", default="")
    ap.add_argument("--n", type=int, default=6400)
    ap.add_argument("--num-classes", type=int, default=17)
    ap.add_argument("--feat-dim", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(Path(args.index).read_text())
    tokens = [t for s in scenes for t in s["tokens"]]

    enc = None
    if args.encoder == "gf2":
        enc = build_gf2_encoder(args.gf2_ckpt, args.gf2_config)

    done = 0
    for tok in tokens:
        path = out / f"{tok}.pt"
        if path.exists():
            continue
        state = enc(tok) if enc else encode_random(tok, args.n, args.num_classes, args.feat_dim)
        torch.save(state, path)
        done += 1
        if done % 500 == 0:
            print(f"{done} cached...")
    print(f"done: {done} new, {len(tokens)} total -> {out}")


if __name__ == "__main__":
    main()
