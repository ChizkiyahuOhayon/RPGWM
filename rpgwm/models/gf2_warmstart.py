"""Partial warm-start of GaussianEncoder from the official GaussianFormer-2
checkpoint (Prob-64, config/prob/nuscenes_gs6400.py — R101-DCN @ 1600x864,
SurroundOcc supervision).

What transfers (decision B1(a), plan §2.0):
  lifter.anchor / lifter.instance_feature -> first 4000 slots (xyz converted
      from the checkpoint's pc_range into ours; remaining slots keep the grid
      prior),
  encoder.anchor_encoder.*                -> anchor_embed.*,
  per decoder block d (their flat `encoder.layers.{i}` under the operation
  order [identity, deformable, add, norm, identity, ffn, add, norm, identity,
  spconv, add, norm, identity, ffn, add, norm, refine] x num_decoder):
      layers.{17d+1}  DeformableFeatureAggregation -> blocks.{d}.deformable
      layers.{17d+3}  LN                           -> blocks.{d}.norm1
      layers.{17d+5}  AsymmetricFFN                -> blocks.{d}.ffn1
      layers.{17d+7}  LN                           -> blocks.{d}.norm2
      layers.{17d+9}  SparseConv3D                 -> blocks.{d}.self_interact
                      when the encoder was built with self_interact='spconv'
                      (server path, 100% op coverage); skipped for the kNN
                      fallback (zero-init output => identity at t=0 — sound
                      because the op is residual: operation order
                      identity->spconv->add->norm)
      layers.{17d+11} LN                           -> blocks.{d}.norm3
      layers.{17d+13} AsymmetricFFN                -> blocks.{d}.ffn2
      layers.{17d+15} LN                           -> blocks.{d}.norm4
      layers.{17d+16} RefinementModuleV2           -> blocks.{d}.refine
What never transfers: img_backbone/img_neck (R101-DCN vs our R50-FPN — the
weak-start cost accepted in B1(a)), the lifter initializer CNN, spconv ops.

The loader is strict about shapes: a key that exists on both sides with a
different shape is an ERROR (config drift), not a silent skip.
"""
from __future__ import annotations

import torch

from .encoder import GaussianEncoder, safe_inverse_sigmoid, safe_sigmoid

GF2_PC_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)   # SurroundOcc range
OPS_PER_DECODER = 17
BLOCK_OFFSETS = {  # op index within a decoder -> our submodule name
    1: "deformable", 3: "norm1", 5: "ffn1", 7: "norm2",
    11: "norm3", 13: "ffn2", 15: "norm4", 16: "refine",
}
SKIP_PREFIXES = ("img_backbone.", "img_neck.", "head.", "lifter.initializer",
                 "future_decoder.")


def build_key_map(encoder: GaussianEncoder) -> dict[str, str]:
    """checkpoint key -> our key, for every transferable parameter."""
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


def convert_lifter_anchor(ck_anchor: torch.Tensor, encoder: GaussianEncoder,
                          src_pc_range=GF2_PC_RANGE):
    """Re-express checkpoint anchors (normalized in src_pc_range) in our
    pc_range. Slots landing outside our (smaller) range are clamped by the
    safe inverse sigmoid — they re-localize within the first refinements.
    Returns (converted anchors, in-our-range mask before clamping)."""
    lo_s = ck_anchor.new_tensor(src_pc_range[:3])
    hi_s = ck_anchor.new_tensor(src_pc_range[3:])
    xyz_m = safe_sigmoid(ck_anchor[..., :3]) * (hi_s - lo_s) + lo_s
    lo_d = ck_anchor.new_tensor(encoder.codec.pc_range[:3])
    hi_d = ck_anchor.new_tensor(encoder.codec.pc_range[3:])
    unit = (xyz_m - lo_d) / (hi_d - lo_d)
    in_range = ((unit > 0.0) & (unit < 1.0)).all(-1)
    return torch.cat([safe_inverse_sigmoid(unit), ck_anchor[..., 3:]], dim=-1), in_range


def resample_fill(converted: torch.Tensor, n_fill: int, encoder: GaussianEncoder,
                  jitter_m: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Fill the remaining slots by resampling the GF-2 anchor EMPIRICAL
    distribution (with a small metric xyz jitter) instead of mixing in a 37.5%
    heterogeneous cold-start population (decision 2026-07-16, item 3c)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(converted.shape[0], (n_fill,), generator=g)
    fill = converted[idx].clone()
    lo = fill.new_tensor(encoder.codec.pc_range[:3])
    hi = fill.new_tensor(encoder.codec.pc_range[3:])
    xyz_m = safe_sigmoid(fill[:, :3]) * (hi - lo) + lo
    xyz_m = xyz_m + torch.randn(n_fill, 3, generator=g) * jitter_m
    fill[:, :3] = safe_inverse_sigmoid((xyz_m - lo) / (hi - lo))
    return fill


def load_gf2_partial(encoder: GaussianEncoder, ck_state: dict,
                     src_pc_range=GF2_PC_RANGE, min_coverage: float = 0.9,
                     verbose: bool = True) -> dict:
    """Copy every transferable tensor; returns a coverage report.

    min_coverage: fraction of OUR transferable keys that must actually be
    found in the checkpoint — below it we raise (wrong checkpoint / drift)
    instead of training silently from near-scratch.
    """
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

    # -- lifter anchors: warm slots + empirical resample for the rest -------
    anchor_slots, kept_frac, fill_n = 0, None, 0
    if "lifter.anchor" in ck_state:
        ck_anchor = ck_state["lifter.anchor"]
        if ck_anchor.shape[-1] != encoder.codec.dim:
            mismatched.append(("lifter.anchor", tuple(ck_anchor.shape),
                               (encoder.num_slots, encoder.codec.dim)))
        else:
            anchor_slots = min(ck_anchor.shape[0], encoder.num_slots)
            converted, in_range = convert_lifter_anchor(ck_anchor[:anchor_slots],
                                                        encoder, src_pc_range)
            kept_frac = round(float(in_range.float().mean()), 4)
            bank = ours["anchor"].clone()
            bank[:anchor_slots] = converted
            fill_n = encoder.num_slots - anchor_slots
            if fill_n > 0:
                bank[anchor_slots:] = resample_fill(converted, fill_n, encoder)
            new_state["anchor"] = bank
            loaded.append("anchor")
    if "lifter.instance_feature" in ck_state:
        ck_feat = ck_state["lifter.instance_feature"]
        if ck_feat.shape[-1] == ours["instance_feature"].shape[-1]:
            feat = ours["instance_feature"].clone()
            n = min(ck_feat.shape[0], feat.shape[0])
            feat[:n] = ck_feat[:n]
            new_state["instance_feature"] = feat
            loaded.append("instance_feature")

    unexpected = [k for k in ck_state
                  if k not in key_map and not k.startswith(SKIP_PREFIXES)
                  and not k.startswith(("lifter.",))]

    if mismatched:
        lines = "\n".join(f"  {k}: ckpt{s} vs ours{d}" for k, s, d in mismatched)
        raise RuntimeError(f"GF2 warm-start shape mismatches (config drift?):\n{lines}")

    coverage = len(loaded) / max(len(key_map) + 2, 1)  # +2: anchor & inst feat
    report = {
        "loaded": len(loaded), "transferable": len(key_map) + 2,
        "coverage": round(coverage, 4), "anchor_slots_warmstarted": anchor_slots,
        "anchor_kept_in_range_frac": kept_frac,   # pre-clamp, decision 3(a)
        "anchor_slots_resampled": fill_n,          # empirical fill, decision 3(c)
        "missing_in_ckpt": len(missing),
        "unexpected_ckpt_keys_sample": sorted(unexpected)[:10],
    }
    if coverage < min_coverage:
        raise RuntimeError(
            f"GF2 warm-start coverage {coverage:.1%} < required {min_coverage:.0%}; "
            f"missing e.g. {missing[:5]} — wrong checkpoint or naming drift.")

    encoder.load_state_dict(new_state, strict=False)
    if verbose:
        print(f"[gf2_warmstart] {report}")
    return report
