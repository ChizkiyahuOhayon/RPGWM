#!/usr/bin/env python3
"""Stage-B trainer: frozen encoder (cached Gaussian states) -> train the
rollout operator W against future Occ3D labels through differentiable
splatting. Gate-1 comparator (copy-last-frame) is evaluated with the exact
same splat/score path.

Single GPU:   python scripts/train.py --config configs/gate1_mini.yaml
2×A40 DDP:    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
                  scripts/train.py --config configs/gate1_mini.yaml
CPU smoke:    python scripts/train.py --config configs/smoke_cpu.yaml --max-steps 3

Every run writes outputs/<run_name>/{report.json, ckpt_last.pt, env.txt}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpgwm.data.nuscenes_occ import OccSequenceDataset, SyntheticSequenceDataset  # noqa: E402
from rpgwm.eval.forecast import FREE_CLASS, forecast_scores  # noqa: E402
from rpgwm.losses import occupancy_recon_loss  # noqa: E402
from rpgwm.models.gaussians import GaussianState  # noqa: E402
from rpgwm.models.rollout import RolloutW  # noqa: E402
from rpgwm.models.splat import VoxelGrid, splat_occupancy, transform_to_future_ego  # noqa: E402


# --------------------------------------------------------------------------
def setup_ddp():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = torch.distributed.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, int(os.environ["WORLD_SIZE"])
    return 0, 1


def state_from_batch(batch, device) -> GaussianState:
    return GaussianState(*(batch[f"state_{k}"].to(device) for k in
                           ("mu", "log_scale", "quat", "opacity", "sem", "feat")))


def rollout_losses(model, state, batch, grid, device):
    """Full K-step rollout -> per-step splat -> recon loss vs future labels."""
    actions = batch["actions"].to(device)
    frames = model.rollout(state, actions) if not hasattr(model, "module") \
        else model.module.rollout(state, actions)
    total, per_step = 0.0, []
    for k, frame in enumerate(frames):
        rot = batch["ego_rot"][:, k].to(device)
        trans = batch["ego_trans"][:, k].to(device)
        moved = transform_to_future_ego(frame, rot, trans)
        occ_prob, _, sem_logit = splat_occupancy(moved, grid)
        loss = occupancy_recon_loss(occ_prob, sem_logit,
                                    batch["labels"][:, k].to(device),
                                    batch["masks"][:, k].to(device))
        total = total + loss
        per_step.append(loss.detach())
    return total / len(frames), per_step


@torch.no_grad()
def evaluate(model, loader, grid, device, max_batches=50):
    """Score the trained rollout AND the copy-last-frame baseline through the
    identical splat/score path (Gate-1 comparison)."""
    model.eval()
    core = model.module if hasattr(model, "module") else model
    agg = {"model": [], "baseline": []}
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        state = state_from_batch(batch, device)
        frames = core.rollout(state, batch["actions"].to(device))
        for name, seq in (("model", frames),
                          ("baseline", [state] * len(frames))):  # copy-last-frame
            preds, gts, masks = [], [], []
            for k, frame in enumerate(seq):
                rot = batch["ego_rot"][:, k].to(device)
                trans = batch["ego_trans"][:, k].to(device)
                moved = transform_to_future_ego(frame, rot, trans)
                _, occ_hard, sem_logit = splat_occupancy(moved, grid)
                preds.append((occ_hard[0].cpu(), sem_logit[0].argmax(-1).cpu()))
                gts.append(batch["labels"][0, k])
                masks.append(batch["masks"][0, k])
            agg[name].append(forecast_scores(preds, gts, masks)["avg"])
    model.train()
    out = {}
    for name, rows in agg.items():
        if rows:
            out[name] = {m: sum(r[m] for r in rows) / len(rows) for m in ("iou", "miou")}
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="override for smoke runs")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rank, world = setup_ddp()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 0) + rank)

    out_dir = Path("outputs") / cfg["run_name"]
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        (out_dir / "env.txt").write_text(
            f"commit={commit}\ntorch={torch.__version__}\nworld={world}\n"
            f"device={device}\nconfig={args.config}\n")

    # -- data ---------------------------------------------------------------
    dcfg = cfg["data"]
    if dcfg.get("synthetic", False):
        ds = SyntheticSequenceDataset(**dcfg.get("synthetic_kwargs", {}))
        val = SyntheticSequenceDataset(**{**dcfg.get("synthetic_kwargs", {}), "seed": 1})
    else:
        ds = OccSequenceDataset(dcfg["index_train"], dcfg["gaussian_cache"],
                                dcfg["occ3d_root"], dcfg["future_frames"])
        val = OccSequenceDataset(dcfg["index_val"], dcfg["gaussian_cache"],
                                 dcfg["occ3d_root"], dcfg["future_frames"])
    tcfg = cfg["train"]
    sampler = DistributedSampler(ds) if world > 1 else None
    loader = DataLoader(ds, batch_size=tcfg.get("micro_batch_per_gpu", 1),
                        shuffle=(sampler is None), sampler=sampler,
                        num_workers=tcfg.get("num_workers", 4), drop_last=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)

    # -- model ----------------------------------------------------------------
    rcfg = cfg["model"]["rollout"]
    model = RolloutW(dim=rcfg["dim"], layers=rcfg["layers"], heads=rcfg["heads"],
                     knn=rcfg["knn"], num_classes=cfg["model"]["num_classes"],
                     feat_dim=cfg["model"]["feat_dim"]).to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg.get("weight_decay", 0.01))
    grid = VoxelGrid(cfg["grid"]["xyz_min"], cfg["grid"]["xyz_max"],
                     cfg["grid"]["resolution"])
    use_bf16 = tcfg.get("precision", "fp32") == "bf16" and device == "cuda"

    # -- loop -----------------------------------------------------------------
    accum = tcfg.get("grad_accum", 1)
    step, t0, running = 0, time.time(), []
    max_steps = args.max_steps or tcfg.get("max_steps", 10 ** 9)
    for epoch in range(tcfg.get("epochs", 1)):
        if sampler:
            sampler.set_epoch(epoch)
        for batch in loader:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                loss, _ = rollout_losses(model, state_from_batch(batch, device),
                                         batch, grid, device)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            running.append(loss.item())
            step += 1
            if rank == 0 and step % tcfg.get("log_every", 50) == 0:
                print(f"epoch {epoch} step {step} loss {sum(running)/len(running):.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                running = []
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # -- eval + report ----------------------------------------------------------
    if rank == 0:
        scores = evaluate(model, val_loader, grid, device,
                          max_batches=tcfg.get("eval_batches", 50))
        gate1 = (scores.get("model", {}).get("miou", 0)
                 > scores.get("baseline", {}).get("miou", 0))
        report = {"run": cfg["run_name"], "steps": step, "scores": scores,
                  "gate1_beats_copy_last_frame": bool(gate1),
                  "wall_seconds": round(time.time() - t0, 1)}
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        core = model.module if hasattr(model, "module") else model
        torch.save({"model": core.state_dict(), "config": cfg}, out_dir / "ckpt_last.pt")
        print(json.dumps(report, indent=2))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
