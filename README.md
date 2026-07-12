# RP-GWM — Reliability-Partitioned Gaussian World Model

CVPR 2027 project. One identity-preserving sparse Gaussian state lives from
perception through action-conditioned rollout to the planner; per-Gaussian
unreliability ρ is learned from *realized* rollout errors and the planner
consumes trusted/untrusted geometry differently.

Research plan: `../RPGWM_CVPR2027_Research_Plan_v4.md`
Idea provenance: `../ideaspark_run/gaussian-occ-wm-e2e/phase4/idea.std.zh.md`
Server instructions: `SERVER_RUNBOOK.md` (2×A40, GPU 0/1)

## Layout
```
rpgwm/models/gaussians.py     GaussianState — the single shared state (slots = identity)
rpgwm/models/rollout.py       M1: rollout operator W (kNN attn + action cross-attn), Eq. 1
rpgwm/models/splat.py         differentiable Gaussian→voxel splatting + future-ego transform
rpgwm/models/reliability.py   M2: realized error e_i, quantile normalizer, ρ head, ECE,
                              partition/inflation (Eq. 2/4); analytic Σ_prop (BeliefGauss legacy)
rpgwm/losses.py               recon + ρ regression + plan-sufficiency (Eq. 5/6)
rpgwm/eval/forecast.py        Occ3D masked IoU/mIoU forecasting protocol (must be
                              cross-checked vs official OccWorld eval before use)
tests/                        CPU unit tests — ALL must pass before anything ships to GPU
configs/gate1_mini.yaml       Gate-1 experiment spec
```

## Dev loop (remote-GPU discipline)
1. Write/modify code locally, `.venv/bin/python -m pytest tests/ -q` must be green.
2. Push; on the A40 server follow `SERVER_RUNBOOK.md` verbatim.
3. Every experiment = one config + one command; results auto-packed with git hash.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q   # 26 passed
```
