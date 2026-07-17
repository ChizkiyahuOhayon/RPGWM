# RP-GWM — Reliability-Partitioned Gaussian World Model

CVPR 2027 project. One identity-preserving sparse Gaussian state lives from
perception through action-conditioned rollout to the planner; per-Gaussian
unreliability ρ is learned from *realized* rollout errors and the planner
consumes trusted/untrusted geometry differently.

Research plan: `docs/RPGWM_CVPR2027_Research_Plan_v5.md` (v4/v3.1 kept for history)
Server instructions: `SERVER_RUNBOOK.md` (2×A40, GPU 0/1) — follow it verbatim, top to bottom.

```bash
# Server quickstart (details + acceptance criteria in SERVER_RUNBOOK.md):
git clone --recurse-submodules git@github.com:ChizkiyahuOhayon/RPGWM.git
```

## Layout
```
rpgwm/models/gaussians.py     GaussianState — the single shared state (slots = identity)
rpgwm/models/encoder.py       §2.0 perception encoder: R50+FPN + 4 GF-2-mirrored
                              refinement blocks over N fixed slots; streaming rigid
                              warp (in-place re-seeding keeps slot identity)
rpgwm/models/gf2_warmstart.py LOTTERY warm-start: only the Gaussian refinement blocks
                              transfer from GF-2 Prob-64 (all released ckpts are
                              R101-DCN@1600x864); acceptance = stage-A 1/4-split A/B
rpgwm/models/rollout.py       M1: rollout operator W (kNN attn + action cross-attn), Eq. 1
rpgwm/models/splat.py         differentiable Gaussian→voxel splatting + future-ego transform
rpgwm/models/reliability.py   M2: realized error e_i, quantile normalizer, ρ head, ECE,
                              partition/inflation (Eq. 2/4); analytic Σ_prop (BeliefGauss legacy)
rpgwm/losses.py               recon + ρ regression + plan-sufficiency (Eq. 5/6)
rpgwm/eval/forecast.py        Occ3D masked IoU/mIoU forecasting protocol (must be
                              cross-checked vs official OccWorld eval before use)
rpgwm/data/nuscenes_occ.py    sequence dataset (cached Gaussian states + Occ3D labels);
                              SyntheticSequenceDataset mirrors the contract for CPU tests
scripts/build_index.py        one-off (server): devkit -> plain JSON scene index
scripts/dump_gaussians.py     one-off (server): encoder inference -> per-token state cache
                              (--encoder random for plumbing tests; gf2 hook = RUNBOOK step 2)
scripts/train.py              stage-B trainer (DDP-ready, bf16, grad-accum, report.json,
                              Gate-1 eval vs copy-last-frame through the same splat path)
scripts/crosscheck_eval.py    our protocol vs official OccWorld eval on identical dumps
scripts/train_encoder.py      stage-A trainer (official GF-2 loss recipe, warmstart A/B,
                              measured iter/s -> epoch-budget block in report.json)
scripts/ag_probe_vista.py     A-G teacher probe: Vista block shapes + s/clip + cache size
tests/                        CPU unit tests — ALL must pass before anything ships to GPU
configs/gate1_mini.yaml       Gate-1 experiment spec
configs/stage_a.yaml          stage-A encoder training (RUNBOOK §2.5)
configs/smoke_cpu.yaml        <1 min full-loop CPU smoke (run before every push)
```

## Dev loop (remote-GPU discipline)
1. Write/modify code locally, `.venv/bin/python -m pytest tests/ -q` must be green.
2. Push; on the A40 server follow `SERVER_RUNBOOK.md` verbatim.
3. Every experiment = one config + one command; results auto-packed with git hash.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q   # 55 passed
```
