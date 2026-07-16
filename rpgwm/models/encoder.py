"""§2.0 perception encoder (background, not a contribution).

R50 + FPN over 6 cameras -> 4 attention refinement blocks maintaining a FIXED
bank of N Gaussian slots. Across frames the previous slots are rigidly warped
into the current ego frame first, then refined against the new images
(S2GO/GaussianWorld-style streaming). Slots are never inserted, deleted, or
permuted; a slot whose content leaves the perception range is re-seeded IN
PLACE from the anchor prior (same index — positional identity is the contract
everything downstream relies on).

Architecture mirrors GaussianFormer-2 `config/prob/nuscenes_gs6400.py` at the
parameter level so the official checkpoint partially warm-starts us
(rpgwm/models/gf2_warmstart.py):
  - anchor vector   [x y z | s1 s2 s3 | quat(4) | opacity | sem logits(C)],
                    xyz inverse-sigmoid-normalized in pc_range, scales
                    pre-sigmoid in scale_range — GF-2's exact parametrization;
  - GaussianAnchorEmbed        == SparseGaussian3DEncoder      (transferable)
  - DeformableImageAggregation == DeformableFeatureAggregation (transferable;
                    we implement its documented pure-PyTorch sampling path —
                    grid_sample + masked softmax over cams×levels×points —
                    with identical parameter shapes, no CUDA op)
  - AsymFFN                    == AsymmetricFFN                 (transferable)
  - RefineHead                 == SparseGaussian3DRefinementModuleV2 (transf.)
  - SlotSelfAttention REPLACES their SparseConv3D op (spconv dependency;
                    weights intentionally not transferred — output projection
                    zero-init so a warm-started block is undisturbed at t=0).

Block operation order copies GF-2: deformable -> add -> norm | ffn -> add ->
norm | self-interact -> add -> norm | ffn -> add -> norm | refine.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussians import GaussianState, quat_multiply, quat_to_rotmat
from .rollout import NeighborAttention, knn_indices
from .splat import rotmat_to_quat

LOGIT_EPS = 1e-4  # GF-2 LOGIT_MAX = 0.9999


def safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.clamp(min=math.log(LOGIT_EPS / (1 - LOGIT_EPS)),
                                 max=math.log((1 - LOGIT_EPS) / LOGIT_EPS)))


def safe_inverse_sigmoid(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
    return torch.log(p) - torch.log1p(-p)


def linear_relu_ln(embed_dims: int, in_loops: int, out_loops: int,
                   input_dims: int | None = None) -> list[nn.Module]:
    """GF-2 utils.linear_relu_ln, byte-identical layer layout (state-dict
    indices must match for warm-start)."""
    if input_dims is None:
        input_dims = embed_dims
    layers: list[nn.Module] = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class ElementScale(nn.Module):
    """mmcv Scale with a vector: y = x * scale (parameter name must be
    `scale` for warm-start)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


# ---------------------------------------------------------------------------
# anchor <-> GaussianState conversion
# ---------------------------------------------------------------------------
class AnchorCodec:
    """Raw GF-2 anchor vector <-> physical quantities.

    Layout: [xyz(3, inv-sigmoid in pc_range) | scale(3, pre-sigmoid in
    scale_range) | quat(4, w-x-y-z) | opacity(1, pre-sigmoid) | sem(C logits)].
    """

    def __init__(self, pc_range, scale_range, num_classes: int):
        self.pc_range = tuple(float(v) for v in pc_range)
        self.scale_range = tuple(float(v) for v in scale_range)
        self.num_classes = num_classes
        self.dim = 3 + 3 + 4 + 1 + num_classes

    def _range_tensors(self, ref: torch.Tensor):
        lo = ref.new_tensor(self.pc_range[:3])
        hi = ref.new_tensor(self.pc_range[3:])
        return lo, hi

    def xyz(self, anchor: torch.Tensor) -> torch.Tensor:
        lo, hi = self._range_tensors(anchor)
        return safe_sigmoid(anchor[..., :3]) * (hi - lo) + lo

    def set_xyz(self, anchor: torch.Tensor, xyz_m: torch.Tensor) -> torch.Tensor:
        lo, hi = self._range_tensors(anchor)
        unit = (xyz_m - lo) / (hi - lo)
        return torch.cat([safe_inverse_sigmoid(unit), anchor[..., 3:]], dim=-1)

    def scales_m(self, anchor: torch.Tensor) -> torch.Tensor:
        s0, s1 = self.scale_range
        return s0 + (s1 - s0) * safe_sigmoid(anchor[..., 3:6])

    def in_range_mask(self, xyz_m: torch.Tensor, margin: float = 1e-3) -> torch.Tensor:
        lo, hi = self._range_tensors(xyz_m)
        unit = (xyz_m - lo) / (hi - lo)
        return ((unit > margin) & (unit < 1.0 - margin)).all(-1)

    def to_state(self, anchor: torch.Tensor, feat: torch.Tensor) -> GaussianState:
        return GaussianState(
            mu=self.xyz(anchor),
            log_scale=torch.log(self.scales_m(anchor)),
            quat=F.normalize(anchor[..., 6:10], dim=-1),
            opacity=safe_sigmoid(anchor[..., 10]),
            sem=anchor[..., 11:11 + self.num_classes],
            feat=feat,
        )


def make_grid_anchors(num_slots: int, codec: AnchorCodec, resolution=None,
                      seed: int = 0) -> torch.Tensor:
    """Cold-start anchor bank: an even xyz grid over pc_range (GaussianWorld's
    anchor-prior recipe), identity rotation, ~0.5 m scales, opacity 0.1."""
    if resolution is None:
        r = max(2, round((num_slots / 4) ** (1 / 3)))  # z thinner than x/y
        resolution = (2 * r, 2 * r, max(2, num_slots // (4 * r * r)))
    xs = torch.linspace(0.02, 0.98, resolution[0])
    ys = torch.linspace(0.02, 0.98, resolution[1])
    zs = torch.linspace(0.02, 0.98, resolution[2])
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    unit = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    if unit.shape[0] < num_slots:  # top up with low-discrepancy random points
        g = torch.Generator().manual_seed(seed)
        extra = torch.rand(num_slots - unit.shape[0], 3, generator=g) * 0.96 + 0.02
        unit = torch.cat([unit, extra])
    idx = torch.linspace(0, unit.shape[0] - 1, num_slots).long()
    xyz = safe_inverse_sigmoid(unit[idx])

    s0, s1 = codec.scale_range
    target = min(max(0.5, s0 + 1e-3), s1 - 1e-3)  # ~0.5 m std
    scale = safe_inverse_sigmoid(torch.full((num_slots, 3), (target - s0) / (s1 - s0)))
    quat = torch.zeros(num_slots, 4)
    quat[:, 0] = 1.0
    opacity = safe_inverse_sigmoid(torch.full((num_slots, 1), 0.1))
    sem = torch.zeros(num_slots, codec.num_classes)
    return torch.cat([xyz, scale, quat, opacity, sem], dim=-1)


# ---------------------------------------------------------------------------
# image backbone
# ---------------------------------------------------------------------------
class CameraFPN(nn.Module):
    """ResNet-50 + FPN over all cameras. Levels: strides 8/16/32/64 (GF-2 uses
    neck start_level=1, i.e. also drops the stride-4 map). Weights come from
    ImageNet/FCOS3D on the server; random init here (see gate1_mini.yaml note
    on the weak-start backbone)."""

    STRIDES = (8, 16, 32, 64)
    STAGE_CHANNELS = {"resnet18": [64, 128, 256, 512],
                      "resnet34": [64, 128, 256, 512],
                      "resnet50": [256, 512, 1024, 2048],
                      "resnet101": [256, 512, 1024, 2048]}

    def __init__(self, out_dim: int = 128, backbone: str = "resnet50",
                 backbone_weights: str | None = None):
        super().__init__()
        import torchvision
        from torchvision.ops import FeaturePyramidNetwork

        weights = None
        if backbone_weights == "imagenet":
            weights = {"resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
                       "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                       }[backbone]
        net = getattr(torchvision.models, backbone)(weights=weights)
        if backbone_weights and backbone_weights not in (None, "imagenet"):
            sd = torch.load(backbone_weights, map_location="cpu")
            net.load_state_dict(sd.get("state_dict", sd), strict=False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.fpn = FeaturePyramidNetwork(self.STAGE_CHANNELS[backbone], out_dim)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """images [B, cams, 3, H, W] -> 4 maps [B, cams, D, H/s, W/s],
        s = 8, 16, 32, 64."""
        B, N = images.shape[:2]
        x = images.flatten(0, 1)
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        ps = list(self.fpn({"0": c1, "1": c2, "2": c3, "3": c4}).values())
        levels = [ps[1], ps[2], ps[3], F.max_pool2d(ps[3], 1, stride=2)]
        return [lv.reshape(B, N, *lv.shape[1:]) for lv in levels]


# ---------------------------------------------------------------------------
# GF-2 mirrored submodules
# ---------------------------------------------------------------------------
class GaussianAnchorEmbed(nn.Module):
    """Mirror of SparseGaussian3DEncoder (embeds the raw anchor vector)."""

    def __init__(self, embed_dims: int, num_classes: int):
        super().__init__()

        def emb(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.xyz_fc = emb(3)
        self.scale_fc = emb(3)
        self.rot_fc = emb(4)
        self.opacity_fc = emb(1)
        self.semantics_fc = emb(num_classes)
        self.output_fc = emb(embed_dims)
        self.num_classes = num_classes

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        out = (self.xyz_fc(anchor[..., :3]) + self.scale_fc(anchor[..., 3:6])
               + self.rot_fc(anchor[..., 6:10]) + self.opacity_fc(anchor[..., 10:11])
               + self.semantics_fc(anchor[..., 11:11 + self.num_classes]))
        return self.output_fc(out)


class KeyPointsGenerator(nn.Module):
    """Mirror of SparseGaussian3DKeyPointsGenerator: 7 fixed + 6 learnable
    offsets in the Gaussian's own (scaled, rotated) frame."""

    FIX_SCALE = ((0, 0, 0), (0.45, 0, 0), (-0.45, 0, 0), (0, 0.45, 0),
                 (0, -0.45, 0), (0, 0, 0.45), (0, 0, -0.45))

    def __init__(self, embed_dims: int, codec: AnchorCodec,
                 num_learnable_pts: int = 6, learnable_fixed_scale: float = 6.0):
        super().__init__()
        self.codec = codec
        self.num_learnable_pts = num_learnable_pts
        self.learnable_fixed_scale = learnable_fixed_scale
        self.register_buffer("fix_scale", torch.tensor(self.FIX_SCALE, dtype=torch.float),
                             persistent=False)
        self.num_pts = len(self.FIX_SCALE) + num_learnable_pts
        self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)

    def forward(self, anchor: torch.Tensor, instance_feature: torch.Tensor) -> torch.Tensor:
        B, A = anchor.shape[:2]
        offsets = self.fix_scale[None, None].expand(B, A, -1, -1)
        learn = (safe_sigmoid(self.learnable_fc(instance_feature)
                              .reshape(B, A, self.num_learnable_pts, 3)) - 0.5)
        offsets = torch.cat([offsets, learn * self.learnable_fixed_scale], dim=-2)
        key_points = offsets * self.codec.scales_m(anchor).unsqueeze(-2)
        rot = quat_to_rotmat(F.normalize(anchor[..., 6:10], dim=-1)).transpose(-1, -2)
        key_points = (rot[:, :, None] @ key_points.unsqueeze(-1)).squeeze(-1)
        return key_points + self.codec.xyz(anchor).unsqueeze(-2)  # [B, A, P, 3]


class DeformableImageAggregation(nn.Module):
    """Mirror of DeformableFeatureAggregation (use_camera_embed=True,
    residual_mode='none'), pure-PyTorch sampling path. Parameter shapes match
    the official module exactly: camera_encoder, weights_fc, output_proj,
    kps_generator.learnable_fc all warm-start."""

    def __init__(self, embed_dims: int, codec: AnchorCodec, num_groups: int = 4,
                 num_levels: int = 4, num_cams: int = 6):
        super().__init__()
        assert embed_dims % num_groups == 0
        self.embed_dims, self.num_groups = embed_dims, num_groups
        self.group_dims = embed_dims // num_groups
        self.num_levels, self.num_cams = num_levels, num_cams
        self.kps_generator = KeyPointsGenerator(embed_dims, codec)
        self.num_pts = self.kps_generator.num_pts
        self.camera_encoder = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
        self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    @staticmethod
    def project_points(key_points: torch.Tensor, projection: torch.Tensor,
                       image_wh: torch.Tensor):
        """key_points [B, A, P, 3] (ego frame), projection [B, cams, 4, 4]
        ego->image (pixels), image_wh [2] -> normalized uv [B, cams, A, P, 2]
        in [0,1] + validity mask."""
        pts = torch.cat([key_points, torch.ones_like(key_points[..., :1])], -1)
        cam = torch.einsum("bcij,bapj->bcapi", projection, pts)
        depth = cam[..., 2]
        uv = cam[..., :2] / depth.unsqueeze(-1).clamp_min(1e-5)
        uv = uv / image_wh
        mask = ((depth > 1e-5) & (uv[..., 0] > 0) & (uv[..., 0] < 1)
                & (uv[..., 1] > 0) & (uv[..., 1] < 1))
        return uv, mask

    def forward(self, instance_feature, anchor, anchor_embed, feature_maps,
                projection, image_wh):
        B, A = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        uv, proj_mask = self.project_points(key_points, projection, image_wh)

        # weights: [B, A, cams, levels, pts, groups], masked softmax over
        # (cams, levels, pts) jointly — the official DAF-path semantics.
        feat = instance_feature + anchor_embed
        cam_embed = self.camera_encoder(projection[:, :, :3].reshape(B, self.num_cams, 12))
        w = self.weights_fc(feat[:, :, None] + cam_embed[:, None])
        w = w.reshape(B, A, self.num_cams, self.num_levels, self.num_pts, self.num_groups)
        # -1e4 (finite, bf16-safe) instead of -inf: softmax stays NaN-free
        w = w.masked_fill(~proj_mask.permute(0, 2, 1, 3)[..., None, :, None], -1e4)
        w = w.reshape(B, A, -1, self.num_groups).softmax(dim=2)
        # anchors visible in no camera get zero aggregation, not a uniform mix
        any_hit = proj_mask.any(1).any(-1)                      # [B, A]
        w = w * any_hit[..., None, None]
        w = w.reshape(B, A, self.num_cams, self.num_levels, self.num_pts, self.num_groups)

        # sample: per cam & level grid_sample at uv
        grid = uv.reshape(B * self.num_cams, A * self.num_pts, 1, 2) * 2 - 1
        sampled = []
        for lv in range(self.num_levels):
            fm = feature_maps[lv].flatten(0, 1)                 # [B*cams, D, h, w]
            s = F.grid_sample(fm, grid, align_corners=False)    # [B*cams, D, A*P, 1]
            sampled.append(s.squeeze(-1).reshape(B, self.num_cams, self.embed_dims,
                                                 A, self.num_pts))
        feats = torch.stack(sampled, dim=2)                     # [B,cams,L,D,A,P]
        feats = feats.permute(0, 4, 1, 2, 5, 3)                 # [B,A,cams,L,P,D]
        feats = feats.reshape(B, A, self.num_cams, self.num_levels, self.num_pts,
                              self.num_groups, self.group_dims)
        out = (w.unsqueeze(-1) * feats).sum(dim=(2, 3, 4)).reshape(B, A, self.embed_dims)
        return self.output_proj(out)


class AsymFFN(nn.Module):
    """Mirror of AsymmetricFFN (prob config: add_identity=False, pre_norm=None,
    feedforward = 4x). State-dict layout: layers.0.0 / layers.1."""

    def __init__(self, embed_dims: int, expansion: int = 4, drop: float = 0.1):
        super().__init__()
        hidden = embed_dims * expansion
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, hidden), nn.ReLU(inplace=True),
                          nn.Dropout(drop)),
            nn.Linear(hidden, embed_dims),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.layers(x)


class SlotSelfAttention(nn.Module):
    """Replaces GF-2's SparseConv3D op (spconv). kNN attention over slot
    centers; output zero-init so a warm-started block starts undisturbed."""

    def __init__(self, embed_dims: int, codec: AnchorCodec, knn: int = 16,
                 heads: int = 4):
        super().__init__()
        self.codec = codec
        self.knn = knn
        self.attn = NeighborAttention(embed_dims, heads)
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.attn.proj.bias)

    def forward(self, instance_feature: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        mu = self.codec.xyz(anchor).detach()
        nbr = knn_indices(mu, min(self.knn, anchor.shape[1] - 1))
        return self.attn(instance_feature, nbr)


class RefineHead(nn.Module):
    """Mirror of SparseGaussian3DRefinementModuleV2: bounded xyz delta
    (unit_xyz box), scale/rot/opacity/semantics REPLACED outright."""

    def __init__(self, embed_dims: int, codec: AnchorCodec, unit_xyz=(4.0, 4.0, 1.0)):
        super().__init__()
        self.codec = codec
        out_dim = codec.dim
        self.register_buffer("unit_xyz", torch.tensor(unit_xyz, dtype=torch.float),
                             persistent=False)
        self.layers = nn.Sequential(*linear_relu_ln(embed_dims, 2, 2),
                                    nn.Linear(embed_dims, out_dim),
                                    ElementScale(out_dim))

    def forward(self, instance_feature, anchor, anchor_embed):
        out = self.layers(instance_feature + anchor_embed)
        delta_xyz = (2 * safe_sigmoid(out[..., :3]) - 1.0) * self.unit_xyz
        new_xyz = self.codec.xyz(anchor) + delta_xyz
        rot = F.normalize(out[..., 6:10], dim=-1)
        refined = torch.cat([out[..., :3], out[..., 3:6], rot, out[..., 10:]], dim=-1)
        return self.codec.set_xyz(refined, new_xyz)


class GaussianEncoderBlock(nn.Module):
    """One refinement block, GF-2 operation order."""

    def __init__(self, embed_dims: int, codec: AnchorCodec, num_groups: int,
                 num_levels: int, num_cams: int, knn: int, drop: float = 0.1):
        super().__init__()
        self.deformable = DeformableImageAggregation(embed_dims, codec, num_groups,
                                                     num_levels, num_cams)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn1 = AsymFFN(embed_dims, drop=drop)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.self_interact = SlotSelfAttention(embed_dims, codec, knn)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.ffn2 = AsymFFN(embed_dims, drop=drop)
        self.norm4 = nn.LayerNorm(embed_dims)
        self.refine = RefineHead(embed_dims, codec)

    def forward(self, inst, anchor, anchor_embed, feature_maps, projection, image_wh):
        inst = self.norm1(inst + self.deformable(inst, anchor, anchor_embed,
                                                 feature_maps, projection, image_wh))
        inst = self.norm2(inst + self.ffn1(inst))
        inst = self.norm3(inst + self.self_interact(inst, anchor))
        inst = self.norm4(inst + self.ffn2(inst))
        anchor = self.refine(inst, anchor, anchor_embed)
        return inst, anchor


# ---------------------------------------------------------------------------
# the full encoder
# ---------------------------------------------------------------------------
class GaussianEncoder(nn.Module):
    """images (+ previous slots) -> refined GaussianState with fixed slot count.

    forward(images, projection, image_wh, prev=None, prev2cur=None)
      images     [B, cams, 3, H, W]
      projection [B, cams, 4, 4]  ego -> image-pixel matrices
      image_wh   [2]              (W, H) of the input images
      prev       (anchor [B,N,dim], inst [B,N,D]) from the previous frame
      prev2cur   [B, 4, 4]        previous-ego -> current-ego transform
    returns (GaussianState, (anchor, inst) to stream into the next frame,
             fill_mask [B,N] — slots re-seeded in place this frame)
    """

    def __init__(self, num_slots: int = 6400, embed_dims: int = 128,
                 num_classes: int = 17, feat_dim: int = 256, num_blocks: int = 4,
                 num_cams: int = 6, num_levels: int = 4, num_groups: int = 4,
                 knn: int = 16, pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                 scale_range=(0.01, 3.2), backbone: str = "resnet50",
                 backbone_weights: str | None = None):
        super().__init__()
        self.codec = AnchorCodec(pc_range, scale_range, num_classes)
        self.num_slots = num_slots
        self.backbone = CameraFPN(embed_dims, backbone, backbone_weights)
        self.anchor = nn.Parameter(make_grid_anchors(num_slots, self.codec))
        self.instance_feature = nn.Parameter(torch.zeros(num_slots, embed_dims),
                                             requires_grad=False)  # GF-2 feat_grad=False
        self.anchor_embed = GaussianAnchorEmbed(embed_dims, num_classes)
        self.blocks = nn.ModuleList([
            GaussianEncoderBlock(embed_dims, self.codec, num_groups, num_levels,
                                 num_cams, knn) for _ in range(num_blocks)])
        self.out_feat = nn.Linear(embed_dims, feat_dim)

    # -- streaming rigid transfer ------------------------------------------
    def warp_slots(self, anchor: torch.Tensor, inst: torch.Tensor,
                   prev2cur: torch.Tensor):
        """Rigidly move every slot into the current ego frame; slots landing
        outside pc_range are re-seeded IN PLACE from the anchor prior (index
        kept — never dropped or reordered). Rotations ARE rotated, unlike
        GaussianWorld's warp which leaves quaternions untouched."""
        xyz = self.codec.xyz(anchor)
        new_xyz = torch.einsum("bij,bnj->bni", prev2cur[:, :3, :3], xyz) \
            + prev2cur[:, :3, 3].unsqueeze(1)
        q_rel = rotmat_to_quat(prev2cur[:, :3, :3])
        new_quat = F.normalize(
            quat_multiply(q_rel.unsqueeze(1).expand(-1, anchor.shape[1], -1),
                          anchor[..., 6:10]), dim=-1)
        warped = torch.cat([anchor[..., :3], anchor[..., 3:6], new_quat,
                            anchor[..., 10:]], dim=-1)
        warped = self.codec.set_xyz(warped, new_xyz)

        keep = self.codec.in_range_mask(new_xyz)                        # [B, N]
        prior_a = self.anchor.detach().unsqueeze(0).expand_as(anchor)
        prior_f = self.instance_feature.detach().unsqueeze(0).expand_as(inst)
        warped = torch.where(keep.unsqueeze(-1), warped, prior_a)
        inst = torch.where(keep.unsqueeze(-1), inst, prior_f)
        return warped, inst, ~keep

    def forward(self, images: torch.Tensor, projection: torch.Tensor,
                image_wh: torch.Tensor, prev=None, prev2cur=None):
        B = images.shape[0]
        feature_maps = self.backbone(images)
        if prev is None:
            anchor = self.anchor.unsqueeze(0).expand(B, -1, -1)
            inst = self.instance_feature.unsqueeze(0).expand(B, -1, -1)
            fill_mask = torch.ones(B, self.num_slots, dtype=torch.bool,
                                   device=images.device)
        else:
            assert prev2cur is not None, "streaming needs the ego transform"
            anchor, inst, fill_mask = self.warp_slots(prev[0], prev[1], prev2cur)

        for blk in self.blocks:
            embed = self.anchor_embed(anchor)
            inst, anchor = blk(inst, anchor, embed, feature_maps, projection, image_wh)

        state = self.codec.to_state(anchor, self.out_feat(inst))
        return state, (anchor, inst), fill_mask
