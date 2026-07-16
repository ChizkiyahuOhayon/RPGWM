"""CPU smoke tests for the §2.0 perception encoder skeleton:
shapes/ranges, streaming rigid transfer (exactness + in-place re-seeding),
gradient flow into the image backbone, GF-2 partial warm-start mapping, and
the dataset's history-frame contract."""
import math

import pytest
import torch
import torch.nn.functional as F

from rpgwm.models.encoder import GaussianEncoder, safe_inverse_sigmoid
from rpgwm.models.gaussians import quat_to_rotmat
from rpgwm.models.gf2_warmstart import build_key_map, load_gf2_partial

B, CAMS, NC = 1, 2, 6
H_IMG, W_IMG = 64, 128
PC_RANGE = (-8.0, -8.0, -2.0, 8.0, 8.0, 2.0)


def make_encoder(num_slots=40, embed=32, blocks=2):
    torch.manual_seed(0)
    return GaussianEncoder(
        num_slots=num_slots, embed_dims=embed, num_classes=NC, feat_dim=16,
        num_blocks=blocks, num_cams=CAMS, num_groups=4, knn=8,
        pc_range=PC_RANGE, backbone="resnet18")


def make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(B, CAMS, 3, H_IMG, W_IMG, generator=g)
    # two pinhole cameras at the ego origin looking along +x and -x
    K = torch.eye(4)
    K[0, 0] = K[1, 1] = 50.0
    K[0, 2], K[1, 2] = W_IMG / 2, H_IMG / 2
    def look(sign):
        R = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) \
            * torch.tensor([1.0, 1.0, sign]).view(3, 1)
        E = torch.eye(4)
        E[:3, :3] = R
        return K @ E
    projection = torch.stack([look(1.0), look(-1.0)]).unsqueeze(0)  # [1, 2, 4, 4]
    image_wh = torch.tensor([float(W_IMG), float(H_IMG)])
    return images, projection, image_wh


def test_forward_shapes_and_ranges():
    enc, (images, proj, wh) = make_encoder(), make_inputs()
    state, (anchor, inst), fill = enc(images, proj, wh)
    N = enc.num_slots
    assert state.mu.shape == (B, N, 3)
    assert state.log_scale.shape == (B, N, 3)
    assert state.quat.shape == (B, N, 4)
    assert state.opacity.shape == (B, N)
    assert state.sem.shape == (B, N, NC)
    assert state.feat.shape == (B, N, 16)
    assert anchor.shape == (B, N, enc.codec.dim) and inst.shape[1] == N
    assert fill.all()  # cold start: every slot freshly seeded
    assert (state.opacity >= 0).all() and (state.opacity <= 1).all()
    norms = state.quat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    lo = state.mu.new_tensor(PC_RANGE[:3])
    hi = state.mu.new_tensor(PC_RANGE[3:])
    assert (state.mu >= lo - 1e-4).all() and (state.mu <= hi + 1e-4).all()


def test_streaming_keeps_slot_count_and_order():
    enc, (images, proj, wh) = make_encoder(), make_inputs()
    with torch.no_grad():
        _, prev, _ = enc(images, proj, wh)
        prev2cur = torch.eye(4).unsqueeze(0)
        prev2cur[0, 0, 3] = 1.0  # ego moved 1 m forward
        state2, nxt, fill = enc(make_inputs(seed=1)[0], proj, wh,
                                prev=prev, prev2cur=prev2cur)
    assert nxt[0].shape == prev[0].shape and nxt[1].shape == prev[1].shape
    assert state2.mu.shape[1] == enc.num_slots
    assert fill.dtype == torch.bool and fill.shape == (B, enc.num_slots)


def test_warp_is_exactly_rigid_and_reseeds_in_place():
    enc = make_encoder()
    anchor = enc.anchor.detach().unsqueeze(0).clone()
    inst = torch.randn(1, enc.num_slots, 32)

    # 90° yaw + 1 m forward
    yaw = math.pi / 2
    T = torch.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = 0.0, -1.0, 1.0, 0.0
    T[0, 3] = 1.0
    warped, w_inst, fill = enc.warp_slots(anchor, inst, T.unsqueeze(0))

    xyz_before = enc.codec.xyz(anchor)
    expect = torch.einsum("ij,bnj->bni", T[:3, :3], xyz_before) + T[:3, 3]
    inside = enc.codec.in_range_mask(expect)
    kept = inside[0]
    assert kept.any() and (~kept).any(), "test setup should straddle the range"

    xyz_after = enc.codec.xyz(warped)
    assert torch.allclose(xyz_after[0][kept], expect[0][kept], atol=1e-3)
    # rotations must rotate too (GaussianWorld's warp skips this — we don't)
    R_after = quat_to_rotmat(warped[0, kept][..., 6:10])
    R_expect = T[:3, :3] @ quat_to_rotmat(anchor[0, kept][..., 6:10])
    assert torch.allclose(R_after, R_expect, atol=1e-4)
    # out-of-range slots: re-seeded from the prior at the SAME index
    assert torch.equal(fill[0], ~kept)
    assert torch.allclose(warped[0][~kept], enc.anchor.detach()[~kept])
    assert (w_inst[0][~kept] == 0).all()  # instance-feature prior is zeros
    assert torch.allclose(w_inst[0][kept], inst[0][kept])


def test_gradients_reach_the_backbone():
    enc, (images, proj, wh) = make_encoder(num_slots=24, blocks=1), make_inputs()
    images.requires_grad_(True)
    state, _, _ = enc(images, proj, wh)
    (state.mu.sum() + state.opacity.sum() + state.sem.sum()).backward()
    assert images.grad is not None and images.grad.abs().sum() > 0
    stem_grads = [p.grad for p in enc.backbone.stem.parameters()
                  if p.grad is not None]
    assert stem_grads and any(g.abs().sum() > 0 for g in stem_grads)


def test_gf2_warmstart_mapping_and_anchor_conversion():
    enc = make_encoder()
    key_map = build_key_map(enc)
    assert key_map, "no transferable keys found"
    ours = enc.state_dict()

    g = torch.Generator().manual_seed(7)
    fake = {ck: torch.randn(ours[our].shape, generator=g)
            for ck, our in key_map.items()}
    # lifter: 16 anchors with xyz at the exact center of the SOURCE range
    n_lift = 16
    lift = torch.randn(n_lift, enc.codec.dim, generator=g)
    lift[:, :3] = safe_inverse_sigmoid(torch.full((n_lift, 3), 0.5))
    fake["lifter.anchor"] = lift
    fake["lifter.instance_feature"] = torch.randn(n_lift, 32, generator=g)
    fake["img_backbone.conv1.weight"] = torch.randn(4, 4)  # must be ignored

    report = load_gf2_partial(enc, fake, src_pc_range=(-8, -8, -2, 8, 8, 2),
                              verbose=False)
    assert report["coverage"] == 1.0
    assert report["anchor_slots_warmstarted"] == n_lift

    # a mapped weight really got copied
    probe_ck = "encoder.layers.1.output_proj.weight"
    assert torch.allclose(enc.state_dict()["blocks.0.deformable.output_proj.weight"],
                          fake[probe_ck])
    # center of source range (0,0,0) m must land at (0,0,0) m in our range
    mu0 = enc.codec.xyz(enc.anchor[None])[0, :n_lift]
    assert torch.allclose(mu0, torch.zeros_like(mu0), atol=1e-3)
    # non-xyz attributes copied verbatim
    assert torch.allclose(enc.anchor[:n_lift, 3:], lift[:, 3:])


def test_gf2_warmstart_rejects_shape_drift():
    enc = make_encoder()
    key_map = build_key_map(enc)
    ours = enc.state_dict()
    fake = {ck: torch.randn(ours[our].shape) for ck, our in key_map.items()}
    bad = next(iter(key_map))
    fake[bad] = torch.randn(3, 3, 3)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_gf2_partial(enc, fake, verbose=False)


def test_dataset_history_contract():
    from rpgwm.data.nuscenes_occ import SyntheticSequenceDataset
    ds = SyntheticSequenceDataset(num_items=2, n_gaussians=16, num_classes=NC,
                                  feat_dim=8, future_frames=2, history_frames=3)
    item = ds[0]
    hp = item["hist_prev2cur"]
    assert hp.shape == (3, 4, 4)
    for T in hp:
        R = T[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)  # rigid
        assert torch.allclose(T[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    # actions/labels contract unchanged
    assert item["actions"].shape == (2, 4)
    assert item["labels"].shape[0] == 2
