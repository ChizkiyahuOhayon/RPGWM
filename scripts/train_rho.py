#!/usr/bin/env python3
"""M2 trainer (stage-B second half): frozen rollout W -> realized per-Gaussian
errors -> quantile fit -> train the rho head -> ECE calibration check.

Two passes over the training set (plan §2.2 / stage B):
  Pass 1  measure e_i for every alive predicted Gaussian at every future step;
          fit one QuantileNormalizer per step + the single-step position-noise
          variance for the analytic covariance propagation.
  Pass 2  train RhoHead(feat, Sigma_prop summary, step) -> Q(e_i) with MSE on
          valid slots (alive & GT support).
Then: ECE on the val split; isotonic correction is fitted only if the
reliability diagram is non-monotone (flagged in the report, applied at
inference by consumers).

Single GPU:  CUDA_VISIBLE_DEVICES=0 python scripts/train_rho.py \
                 --config configs/gate1_mini.yaml --ckpt outputs/gate1_mini/ckpt_last.pt
CPU smoke:   python scripts/train_rho.py --config configs/smoke_cpu.yaml \
                 --ckpt outputs/smoke_cpu/ckpt_last.pt --max-batches 2 --epochs 1

Writes outputs/<run_name>_rho/{rho_head.pt, quantiles.pt, report.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpgwm.data.nuscenes_occ import OccSequenceDataset, SyntheticSequenceDataset  # noqa: E402
from rpgwm.models.error_attribution import (fit_single_step_noise,  # noqa: E402
                                            per_gaussian_realized_error)
from rpgwm.models.gaussians import GaussianState  # noqa: E402
from rpgwm.models.reliability import (QuantileNormalizer, RhoHead,  # noqa: E402
                                      expected_calibration_error,
                                      propagated_cov_summary)
from rpgwm.models.rollout import RolloutW  # noqa: E402
from rpgwm.models.splat import VoxelGrid, splat_occupancy, transform_to_future_ego  # noqa: E402
from rpgwm.losses import rho_regression_loss  # noqa: E402


def build_dataset(cfg, val=False):
    dcfg = cfg["data"]
    if dcfg.get("synthetic", False):
        kw = dict(dcfg.get("synthetic_kwargs", {}))
        if val:
            kw["seed"] = kw.get("seed", 0) + 1
        return SyntheticSequenceDataset(**kw)
    return OccSequenceDataset(dcfg["index_val" if val else "index_train"],
                              dcfg["gaussian_cache"], dcfg["occ3d_root"],
                              dcfg["future_frames"])


def state_from_batch(batch, device):
    return GaussianState(*(batch[f"state_{k}"].to(device) for k in
                           ("mu", "log_scale", "quat", "opacity", "sem", "feat")))


@torch.no_grad()
def rollout_and_measure(model, batch, grid, device):
    """Frozen rollout -> per-step (moved frame, realized-error dict)."""
    state = state_from_batch(batch, device)
    frames = model.rollout(state, batch["actions"].to(device))
    centers = grid.centers(device)
    out = []
    for k, frame in enumerate(frames):
        moved = transform_to_future_ego(frame, batch["ego_rot"][:, k].to(device),
                                        batch["ego_trans"][:, k].to(device))
        _, occ_hard, _ = splat_occupancy(moved, grid)
        err = per_gaussian_realized_error(moved, occ_hard,
                                          batch["labels"][:, k].to(device),
                                          batch["masks"][:, k].to(device), centers)
        out.append((moved, err))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="stage-B rollout checkpoint")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=0, help="cap per pass (smoke)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 0))

    out_dir = Path("outputs") / f"{cfg['run_name']}_rho"
    out_dir.mkdir(parents=True, exist_ok=True)

    # frozen rollout
    rcfg = cfg["model"]["rollout"]
    model = RolloutW(dim=rcfg["dim"], layers=rcfg["layers"], heads=rcfg["heads"],
                     knn=rcfg["knn"], num_classes=cfg["model"]["num_classes"],
                     feat_dim=cfg["model"]["feat_dim"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval().requires_grad_(False)

    grid = VoxelGrid(cfg["grid"]["xyz_min"], cfg["grid"]["xyz_max"],
                     cfg["grid"]["resolution"])
    K = rcfg["steps"]
    loader = DataLoader(build_dataset(cfg), batch_size=1, shuffle=True,
                        num_workers=cfg["train"].get("num_workers", 0))
    val_loader = DataLoader(build_dataset(cfg, val=True), batch_size=1, shuffle=False)
    cap = args.max_batches or len(loader)

    # ---------------- pass 1: error statistics --------------------------------
    t0 = time.time()
    qn = QuantileNormalizer(K)
    errs_per_step: list[list[torch.Tensor]] = [[] for _ in range(K)]
    step0_offsets = []
    for bi, batch in enumerate(loader):
        if bi >= cap:
            break
        for k, (_, err) in enumerate(rollout_and_measure(model, batch, grid, device)):
            errs_per_step[k].append(err["e"][err["valid"]].cpu())
            if k == 0:
                step0_offsets.append(err["center_offset"].reshape(-1, 3).cpu())
    n_valid = 0
    for k in range(K):
        pool = torch.cat(errs_per_step[k]) if errs_per_step[k] else torch.zeros(1)
        qn.fit(k, pool)
        n_valid += len(pool)
    noise_var = fit_single_step_noise(torch.cat(step0_offsets)) if step0_offsets \
        else torch.full((3,), 0.04)
    print(f"pass1: {n_valid} valid (gaussian, step) errors; "
          f"single-step pos var {noise_var.tolist()} ({time.time()-t0:.0f}s)")

    # ---------------- pass 2: train the rho head ------------------------------
    head = RhoHead(feat_dim=cfg["model"]["feat_dim"], num_steps=K).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    cov = torch.stack([propagated_cov_summary(k, noise_var) for k in range(K)]).to(device)
    for epoch in range(args.epochs):
        losses = []
        for bi, batch in enumerate(loader):
            if bi >= cap:
                break
            for k, (moved, err) in enumerate(rollout_and_measure(model, batch, grid, device)):
                target = qn.transform(k, err["e"].cpu()).to(device)
                step_ids = torch.full(err["e"].shape, k, dtype=torch.long, device=device)
                rho = head(moved.feat, cov[k], step_ids)
                loss = rho_regression_loss(rho, target, err["valid"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        print(f"epoch {epoch}: rho loss {sum(losses)/max(len(losses),1):.4f}")

    # ---------------- calibration check on val --------------------------------
    head.eval()
    rhos, targets = [], []
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= cap:
                break
            for k, (moved, err) in enumerate(rollout_and_measure(model, batch, grid, device)):
                v = err["valid"]
                if not v.any():
                    continue
                step_ids = torch.full(err["e"].shape, k, dtype=torch.long, device=device)
                rhos.append(head(moved.feat, cov[k], step_ids)[v].cpu())
                targets.append(qn.transform(k, err["e"].cpu())[v.cpu()])
    if rhos:
        rho_all, tgt_all = torch.cat(rhos), torch.cat(targets)
        ece = float(expected_calibration_error(rho_all, tgt_all))
        # monotonicity of the reliability diagram (10 equal-mass bins)
        order = torch.argsort(rho_all)
        bins = torch.tensor_split(tgt_all[order], 10)
        bin_means = torch.tensor([b.mean() for b in bins if len(b)])
        monotone = bool((bin_means[1:] >= bin_means[:-1] - 0.02).all())
    else:
        ece, monotone = float("nan"), False

    torch.save(head.state_dict(), out_dir / "rho_head.pt")
    torch.save({"quantiles": qn.state_dict(), "noise_var": noise_var,
                "num_steps": K}, out_dir / "quantiles.pt")
    report = {"run": f"{cfg['run_name']}_rho", "pass1_valid_errors": n_valid,
              "single_step_pos_var": noise_var.tolist(), "val_ece": ece,
              "reliability_diagram_monotone": monotone,
              "needs_isotonic_correction": (not monotone),
              "wall_seconds": round(time.time() - t0, 1)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
