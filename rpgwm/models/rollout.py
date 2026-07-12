"""M1 — the rollout operator W (Eq. 1 of the plan).

A sparse set transformer that advances all N Gaussian slots by one 0.5 s step,
conditioned on the ego action for that step. Per layer:
  (a) kNN neighbor attention   — each Gaussian attends to its 32 nearest slots
                                 (neighbor table rebuilt every rollout step),
  (b) action cross-attention   — every token attends to a single action token,
  (c) feed-forward.
Heads emit per-slot deltas: position translation, log-scale increment,
incremental rotation (composed via quaternion product), semantic-logit
increment, opacity increment. Slot identity is preserved by construction —
the module never reindexes, inserts, or drops slots.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussians import GaussianState, quat_multiply

ACTION_DIM = 4  # (dx, dy, dyaw, speed) over the 0.5 s step, ego frame


def knn_indices(mu: torch.Tensor, k: int) -> torch.Tensor:
    """mu: [B, N, 3] -> [B, N, k] indices of the k nearest slots (excl. self)."""
    d = torch.cdist(mu, mu)                                   # [B, N, N]
    d.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))          # exclude self
    return d.topk(k, dim=-1, largest=False).indices


class NeighborAttention(nn.Module):
    """Multi-head attention restricted to a precomputed kNN neighborhood."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        K = nbr.shape[-1]
        q, k, v = self.qkv(x).chunk(3, dim=-1)                # each [B, N, D]
        # gather neighbor keys/values: [B, N, K, D]
        idx = nbr.unsqueeze(-1).expand(B, N, K, D)
        k_n = torch.gather(k.unsqueeze(1).expand(B, N, N, D), 2, idx)
        v_n = torch.gather(v.unsqueeze(1).expand(B, N, N, D), 2, idx)
        # split heads
        q = q.view(B, N, self.heads, self.dh).unsqueeze(3)            # [B,N,H,1,dh]
        k_n = k_n.view(B, N, K, self.heads, self.dh).permute(0, 1, 3, 2, 4)
        v_n = v_n.view(B, N, K, self.heads, self.dh).permute(0, 1, 3, 2, 4)
        attn = (q @ k_n.transpose(-1, -2)) / self.dh ** 0.5           # [B,N,H,1,K]
        out = (attn.softmax(-1) @ v_n).squeeze(3)                     # [B,N,H,dh]
        return self.proj(out.reshape(B, N, D))


class ActionCrossAttention(nn.Module):
    """Every slot token attends to the single action token."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], a: [B, D]. One kv token -> attention weight is 1;
        # a learned gate keeps this from being a blind additive bias.
        k, v = self.kv(a).chunk(2, dim=-1)                    # [B, D] each
        gate = torch.sigmoid((self.q(x) * k.unsqueeze(1)).sum(-1, keepdim=True)
                             / x.shape[-1] ** 0.5)            # [B, N, 1]
        return self.proj(gate * v.unsqueeze(1))


class RolloutBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.nbr_attn = NeighborAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.act_attn = ActionCrossAttention(dim)
        self.n3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, nbr, a):
        x = x + self.nbr_attn(self.n1(x), nbr)
        x = x + self.act_attn(self.n2(x), a)
        return x + self.ffn(self.n3(x))


class RolloutW(nn.Module):
    """Advance the full slot set by one step: G_{t+k+1} = W(G_{t+k}, a_{t+k})."""

    MAX_STEP_M = 4.0        # max per-step translation (m); 0.5 s @ ~28.8 km/h lateral bound is generous
    MAX_DLOGS = 0.2         # max per-step log-scale change
    MAX_DROT = 0.2          # max per-step incremental-rotation vector norm (rad-ish)

    def __init__(self, dim: int = 256, layers: int = 4, heads: int = 4,
                 knn: int = 32, num_classes: int = 17, feat_dim: int = 256,
                 extent: float = 40.0):
        super().__init__()
        self.knn = knn
        self.extent = extent
        in_dim = feat_dim + 3 + 3 + 4 + 1 + num_classes
        self.embed = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.action_embed = nn.Sequential(nn.Linear(ACTION_DIM, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([RolloutBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head_mu = nn.Linear(dim, 3)
        self.head_logs = nn.Linear(dim, 3)
        self.head_rot = nn.Linear(dim, 3)   # incremental rotation as a 3-vector
        self.head_sem = nn.Linear(dim, num_classes)
        self.head_op = nn.Linear(dim, 1)
        self.head_feat = nn.Linear(dim, feat_dim)
        # near-identity init: rollout starts as "everything stays put"
        for h in (self.head_mu, self.head_logs, self.head_rot, self.head_sem, self.head_op):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, state: GaussianState, action: torch.Tensor) -> GaussianState:
        """action: [B, 4] = (dx, dy, dyaw, speed) for this 0.5 s step."""
        B, N = state.batch, state.n
        x = self.embed(torch.cat([
            state.feat, state.mu / self.extent, state.log_scale, state.quat,
            state.opacity.unsqueeze(-1), state.sem,
        ], dim=-1))
        a = self.action_embed(action)
        nbr = knn_indices(state.mu.detach(), min(self.knn, N - 1))  # table rebuilt each step
        for blk in self.blocks:
            x = blk(x, nbr, a)
        h = self.norm(x)

        d_mu = torch.tanh(self.head_mu(h)) * self.MAX_STEP_M
        d_logs = torch.tanh(self.head_logs(h)) * self.MAX_DLOGS
        rvec = torch.tanh(self.head_rot(h)) * self.MAX_DROT
        d_quat = torch.cat([torch.ones_like(rvec[..., :1]), 0.5 * rvec], dim=-1)
        new_quat = F.normalize(quat_multiply(d_quat, state.quat), dim=-1)
        new_op = (state.opacity + self.head_op(h).squeeze(-1)).clamp(0.0, 1.0)

        return GaussianState(
            mu=state.mu + d_mu,
            log_scale=state.log_scale + d_logs,
            quat=new_quat,
            opacity=new_op,
            sem=state.sem + self.head_sem(h),
            feat=state.feat + self.head_feat(h),
        )

    def rollout(self, state: GaussianState, actions: torch.Tensor) -> list[GaussianState]:
        """Unroll K steps. actions: [B, K, 4]. Returns [G_{t+1} .. G_{t+K}]."""
        out, cur = [], state
        for k in range(actions.shape[1]):
            cur = self.forward(cur, actions[:, k])
            out.append(cur)
        return out
