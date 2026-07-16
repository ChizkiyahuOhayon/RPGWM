"""OPPORTUNISTIC (lottery-ticket) partial warm-start of GaussianEncoder from
the official GaussianFormer-2 checkpoint. Decision 2026-07-16: warm-start is
no longer a foundation of stage A — the acceptance test is an A/B on a
1/4-split (GF-2 blocks warm vs random init, 1-2 epochs each); whichever wins
is used. This loader exists because trying is free, not because it is owed
anything.

Why it is a lottery, with evidence (discipline: paper + config, conflicts
recorded, config wins):
  - every released nuScenes checkpoint is R101-DCN + FCOS3D @ final_dim
    1600x864 (config/prob/nuscenes_gs6400.py:8,92,102; nuscenes_gs144000.py:
    8,74,83; paper 2412.04384 §4.2 says R101-DCN, raw 900x1600 — the 864 crop
    comes from the data_aug pipeline, conflict recorded); we are R50 @ 256x704
    (decision B1(a); note config/_base_/surroundocc.py:5 does default to
    (704, 256), but no released checkpoint was trained at it),
  - the transferred blocks were trained on stride-4/8/16/32 SECONDFPN features
    of that backbone; ours are stride-8/16/32/64 R50-FPN (our design choice,
    not a mismatch claim) — shapes match, semantics may not. Only the A/B can
    tell.

WHAT TRANSFERS (the ONLY coverage denominator):
  encoder.anchor_encoder.*        -> anchor_embed.*
  per decoder block d, under the flat `encoder.layers.{i}` of operation order
  [identity, deformable, add, norm, identity, ffn, add, norm, identity,
  spconv, add, norm, identity, ffn, add, norm, refine] x num_decoder:
      layers.{17d+1}  DeformableFeatureAggregation -> blocks.{d}.deformable
      layers.{17d+3/7/11/15} LN                    -> blocks.{d}.norm1/2/3/4
      layers.{17d+5/13} AsymmetricFFN              -> blocks.{d}.ffn1/2
      layers.{17d+9}  SparseConv3D                 -> blocks.{d}.self_interact
                                                      (spconv build only)
      layers.{17d+16} RefinementModuleV2           -> blocks.{d}.refine

NOT_TRANSFERRED manifest (explicit, with reasons) is below; total-coverage
numbers over anything else are banned — they were used once to claim a "100%"
that only proved the mapper agrees with itself.
"""
from __future__ import annotations

import torch

from .encoder import GaussianEncoder

OPS_PER_DECODER = 17
BLOCK_OFFSETS = {  # op index within a decoder -> our submodule name
    1: "deformable", 3: "norm1", 5: "ffn1", 7: "norm2",
    11: "norm3", 13: "ffn2", 15: "norm4", 16: "refine",
}

#: Explicitly not transferred — name: reason (with source evidence).
NOT_TRANSFERRED = {
    "img_backbone": "R101-DCN@1600x864 (prob/nuscenes_gs6400.py:8,92,102) vs "
                    "our R50@256x704 (decision B1(a)); backbone starts from "
                    "ImageNet/nuImages instead",
    "img_neck": "SECONDFPN over strides 4/8/16/32 vs our FPN 8/16/32/64 — "
                "our design choice, incompatible by construction",
    "lifter": "NO static xyz anchor bank exists: lifter.anchor holds only "
              "scale/rot/opacity/sem (gaussian_lifter_v2.py:56-73); xyz is "
              "sampled per frame from the image-conditioned depth "
              "distribution (:169-209, :306-307). lifter.random_anchors "
              "[2400,28] is static+trainable but transferring 2400/6400 "
              "slots is the heterogeneous init we rejected (decision 3c). "
              "lifter.instance_feature is frozen zeros (:80-83) — nothing "
              "to transfer",
    "lifter.initializer": "the distribution-based init module (its own "
                          "R101+SECONDFPN, prob/nuscenes_gs6400.py:127-146) — "
                          "we do not carry it; PixelDistributionLoss "
                          "supervises it and is therefore N/A for us",
    "head": "no transferable weights: GaussianHead's only parameter is "
            "empty_scalar (gaussian_head.py:43), disabled by "
            "with_empty=False (prob/nuscenes_gs6400.py:241 analog)",
    "spconv (knn build)": "when the encoder uses the kNN fallback, "
                          "layers.{17d+9} is skipped; zero-init output in a "
                          "residual slot (config op order identity->spconv->"
                          "add->norm, prob/nuscenes_gs6400.py:219-222) keeps "
                          "warm-started blocks undisturbed at t=0",
}

# -- semantic class alignment (evidence-chain based, not palette-based) ------
# GF-2/SurroundOcc label space, from the LABEL PIPELINE: dense grid init to 17
# (free), occupied voxels take SurroundOcc ids (transform_3d.py:502-507);
# class 0 is masked out of supervision (occ_cam_mask = label != 0, :509);
# eval averages ids 1..16 with these names, empty=17 (eval.py:125-132).
GF2_CLASS_NAMES = (
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation")          # ids 0..16; 17 = empty
# Occ3D-nuScenes label space (Occ3D README / annotations.json category list;
# re-assert against annotations.json on the server — RUNBOOK step 1).
OCC3D_CLASS_NAMES = (
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation")          # ids 0..16; 17 = free


def assert_semantic_alignment(src=GF2_CLASS_NAMES, dst=OCC3D_CLASS_NAMES):
    """Name-keyed identity check. If either side's list ever changes, this
    goes red instead of silently mis-permuting semantic logits. Note: src
    id 0 is masked during GF-2 training (transform_3d.py:509) — the
    transferred class-0 column is effectively untrained; recorded, accepted."""
    if len(src) != len(dst):
        raise RuntimeError(f"semantic class count drift: {len(src)} vs {len(dst)}")
    mism = [(i, s, d) for i, (s, d) in enumerate(zip(src, dst)) if s != d]
    if mism:
        raise RuntimeError(
            "semantic class order differs — a name-keyed permutation of the "
            f"sem rows/cols is REQUIRED before transfer: {mism}")


def build_key_map(encoder: GaussianEncoder) -> dict[str, str]:
    """checkpoint key -> our key, Gaussian-encoder blocks ONLY."""
    key_map: dict[str, str] = {}
    ours = dict(encoder.state_dict())

    def adopt(ck_prefix: str, our_prefix: str):
        for k in ours:
            if k.startswith(our_prefix):
                key_map[ck_prefix + k[len(our_prefix):]] = k

    adopt("encoder.anchor_encoder.", "anchor_embed.")
    for d in range(len(encoder.blocks)):
        base = OPS_PER_DECODER * d
        for off, name in BLOCK_OFFSETS.items():
            adopt(f"encoder.layers.{base + off}.", f"blocks.{d}.{name}.")
        if getattr(encoder.blocks[d].self_interact, "IS_GF2_SPCONV", False):
            adopt(f"encoder.layers.{base + 9}.", f"blocks.{d}.self_interact.")
    return key_map


def load_gf2_partial(encoder: GaussianEncoder, ck_state: dict,
                     min_coverage: float = 0.9, verbose: bool = True) -> dict:
    """Copy the transferable block tensors; returns a coverage report whose
    denominator is EXACTLY the block keys above. Shape mismatch on a mapped
    key is an error (config drift), not a skip. min_coverage guards against
    a wrong checkpoint silently degrading to near-random init."""
    assert_semantic_alignment()
    ck_state = ck_state.get("state_dict", ck_state)
    key_map = build_key_map(encoder)
    ours = encoder.state_dict()

    loaded, mismatched, missing = [], [], []
    new_state = {}
    for ck_key, our_key in key_map.items():
        if ck_key not in ck_state:
            missing.append(ck_key)
            continue
        src, dst = ck_state[ck_key], ours[our_key]
        if src.shape != dst.shape:
            mismatched.append((ck_key, tuple(src.shape), tuple(dst.shape)))
            continue
        new_state[our_key] = src.to(dst.dtype)
        loaded.append(our_key)

    if mismatched:
        lines = "\n".join(f"  {k}: ckpt{s} vs ours{d}" for k, s, d in mismatched)
        raise RuntimeError(f"GF2 warm-start shape mismatches (config drift?):\n{lines}")

    coverage = len(loaded) / max(len(key_map), 1)
    report = {
        "loaded": len(loaded), "transferable_block_keys": len(key_map),
        "block_coverage": round(coverage, 4),
        "missing_in_ckpt": len(missing),
        "not_transferred": sorted(NOT_TRANSFERRED),
        "note": "block_coverage counts Gaussian-encoder blocks ONLY; "
                "backbone/FPN/lifter are on the NOT_TRANSFERRED manifest. "
                "Acceptance is the stage-A 1/4-split A/B, not this number.",
    }
    if coverage < min_coverage:
        raise RuntimeError(
            f"GF2 block coverage {coverage:.1%} < required {min_coverage:.0%}; "
            f"missing e.g. {missing[:5]} — wrong checkpoint or naming drift.")

    encoder.load_state_dict(new_state, strict=False)
    if verbose:
        print(f"[gf2_warmstart] {report}")
    return report
