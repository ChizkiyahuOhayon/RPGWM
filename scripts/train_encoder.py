#!/usr/bin/env python3
"""Stage-A trainer: images -> streaming GaussianEncoder -> differentiable
splat -> official GF-2 loss recipe on the current frame's Occ3D labels.

Disciplines (2026-07-16):
  - EVERYTHING config-driven: resolution, backbone, strides live in CameraFPN
    (our design), pc_range, loss recipe, warm-start mode. No hardcoded policy.
  - Encoder stage is NO-INNOVATION: loss = weighted CE(10) + Lovász(1) with
    per-class weights (rpgwm/losses.official_occupancy_loss). Class weights
    come from the config; GF-2's are SurroundOcc-fitted — recompute on Occ3D
    (scripts note) before full runs. PixelDistributionLoss is N/A (see the
    loss docstring), recorded, not substituted.
  - Warm-start = lottery: cfg model.encoder.warmstart in {none, gf2}. The
    acceptance is the 1/4-split A/B (run twice, compare report.json val
    mIoU) — block_coverage is logged but is NOT the gate.
  - Measures iter/s after warmup and projects the epoch budget so §3.2's
    stage-A row can be re-estimated from measurement, not hope.

Single GPU:  CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config configs/stage_a.yaml
2xA40 DDP:   torchrun --nproc_per_node=2 scripts/train_encoder.py --config configs/stage_a.yaml
CPU smoke:   python scripts/train_encoder.py --config configs/stage_a_smoke_cpu.yaml --max-steps 2
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

from rpgwm.data.images import (ImagePipeline, NuScenesImageSequenceDataset,  # noqa: E402
                               SyntheticImageSequenceDataset)
from rpgwm.eval.forecast import FREE_CLASS, binary_iou, semantic_miou  # noqa: E402
from rpgwm.losses import official_occupancy_loss  # noqa: E402
from rpgwm.models.encoder import GaussianEncoder  # noqa: E402
from rpgwm.models.splat import VoxelGrid, splat_occupancy  # noqa: E402


def fail_fast(cfg: dict, config_path: str):
    """60-second discipline: die with the cause before touching a GPU."""
    problems = []
    d = cfg["data"]
    if not d.get("synthetic", False):
        for key in ("index_train", "index_val"):
            if not Path(d[key]).exists():
                problems.append(f"data.{key}: {d[key]} does not exist")
        if not (Path(d["occ3d_root"]) / "gts").is_dir():
            problems.append(f"data.occ3d_root: {d['occ3d_root']}/gts missing")
        if not Path(d["nuscenes_root"]).is_dir():
            problems.append(f"data.nuscenes_root: {d['nuscenes_root']} missing")
    e = cfg["model"]["encoder"]
    if e.get("warmstart", "none") == "gf2" and not Path(e["init_from"]).exists():
        problems.append(f"warmstart=gf2 but init_from missing: {e['init_from']}")
    if problems:
        print(f"[train_encoder] PREFLIGHT FAILED for {config_path}:\n  - "
              + "\n  - ".join(problems), file=sys.stderr)
        sys.exit(2)


def setup_ddp():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        torch.distributed.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo")
        rank = torch.distributed.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, int(os.environ["WORLD_SIZE"])
    return 0, 1


def build_datasets(cfg):
    d = cfg["data"]
    if d.get("synthetic", False):
        kw = dict(d.get("synthetic_kwargs", {}))
        return (SyntheticImageSequenceDataset(**kw),
                SyntheticImageSequenceDataset(**{**kw, "seed": kw.get("seed", 0) + 1}))
    pipe = ImagePipeline(raw_hw=tuple(d["pipeline"]["raw_hw"]),
                         final_hw=tuple(d["pipeline"]["final_hw"]))
    mk = lambda idx: NuScenesImageSequenceDataset(  # noqa: E731
        idx, d["nuscenes_root"], d["occ3d_root"],
        history_frames=d.get("history_frames", 0), pipeline=pipe)
    return mk(d["index_train"]), mk(d["index_val"])


def build_encoder(cfg, device):
    e = cfg["model"]["encoder"]
    enc = GaussianEncoder(
        num_slots=cfg["model"]["num_gaussians"],
        embed_dims=e.get("embed_dims", 128),
        num_classes=cfg["model"]["num_classes"],
        feat_dim=cfg["model"]["feat_dim"],
        num_blocks=e.get("num_blocks", 4),
        num_cams=e.get("num_cams", 6),
        num_levels=e.get("num_levels", 4),
        num_groups=e.get("num_groups", 4),
        knn=e.get("knn", 16),
        pc_range=tuple(cfg["grid"]["xyz_min"]) + tuple(cfg["grid"]["xyz_max"]),
        scale_range=tuple(e.get("scale_range", (0.01, 3.2))),
        backbone=e.get("backbone", "resnet50"),
        backbone_weights=e.get("backbone_weights"),
        self_interact=e.get("self_interact", "knn"),
    ).to(device)
    ws_report = None
    if e.get("warmstart", "none") == "gf2":
        from rpgwm.models.gf2_warmstart import load_gf2_partial
        ck = torch.load(e["init_from"], map_location="cpu")
        ws_report = load_gf2_partial(enc, ck)
    return enc, ws_report


class StreamingEncoder(torch.nn.Module):
    """Wraps the H-history warm-up + current-frame refinement into ONE
    forward, so DDP sees a single forward per backward (multiple raw
    encoder calls per step would break gradient bucketing)."""

    def __init__(self, encoder: GaussianEncoder, history_grad: bool = False):
        super().__init__()
        self.encoder = encoder
        self.history_grad = history_grad

    def forward(self, images, proj, wh, hist_prev2cur=None):
        H = images.shape[1] - 1
        prev = None
        if H > 0:
            ctx = torch.enable_grad() if self.history_grad else torch.no_grad()
            with ctx:
                for h in range(H):
                    _, prev, _ = self.encoder(
                        images[:, h], proj[:, h], wh, prev=prev,
                        prev2cur=None if prev is None else hist_prev2cur[:, h - 1])
            if not self.history_grad:
                prev = tuple(p.detach() for p in prev)
        prev2cur = hist_prev2cur[:, H - 1] if H > 0 else None
        return self.encoder(images[:, H], proj[:, H], wh, prev=prev,
                            prev2cur=prev2cur)


def stream_forward(model, batch, device):
    """model = StreamingEncoder or its DDP wrapper."""
    return model(batch["images"].to(device), batch["projection"].to(device),
                 batch["image_wh"][0].to(device),
                 batch.get("hist_prev2cur", torch.zeros(0)).to(device)
                 if "hist_prev2cur" in batch else None)


@torch.no_grad()
def evaluate(model, loader, grid, device, cfg, max_batches=50):
    model.eval()
    core = model.module if hasattr(model, "module") else model
    ious, mious = [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        state, _, _ = stream_forward(core, batch, device)
        _, occ_hard, sem_logit = splat_occupancy(state, grid)
        gt = batch["label"].to(device)
        mask = batch["mask"].to(device)
        gt_occ = gt != FREE_CLASS
        ious.append(float(binary_iou(occ_hard[0], gt_occ[0], mask[0])))
        pred_label = torch.where(occ_hard[0], sem_logit[0].argmax(-1),
                                 torch.full_like(gt[0], FREE_CLASS))
        miou, _ = semantic_miou(pred_label.cpu(), gt[0].cpu(), mask[0].cpu(),
                                cfg["model"]["num_classes"])
        mious.append(float(miou))
    model.train()
    n = max(len(ious), 1)
    return {"iou": sum(ious) / n, "miou": sum(mious) / n, "batches": len(ious)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    fail_fast(cfg, args.config)
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

    train_ds, val_ds = build_datasets(cfg)
    tcfg = cfg["train"]
    sampler = DistributedSampler(train_ds) if world > 1 else None
    loader = DataLoader(train_ds, batch_size=tcfg.get("micro_batch_per_gpu", 1),
                        shuffle=(sampler is None), sampler=sampler,
                        num_workers=tcfg.get("num_workers", 4), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    encoder, ws_report = build_encoder(cfg, device)
    model = StreamingEncoder(encoder, tcfg.get("history_grad", False))
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg.get("weight_decay", 0.01))
    grid = VoxelGrid(cfg["grid"]["xyz_min"], cfg["grid"]["xyz_max"],
                     cfg["grid"]["resolution"])
    lcfg = cfg.get("loss", {})
    cw = lcfg.get("class_weights")
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device) if cw else None
    use_bf16 = tcfg.get("precision", "fp32") == "bf16" and device == "cuda"

    accum = tcfg.get("grad_accum", 1)
    warmup_iters = tcfg.get("timing_warmup_iters", 5)
    max_steps = args.max_steps or tcfg.get("max_steps", 10 ** 9)
    step, losses, t_timed, timed_iters = 0, [], 0.0, 0
    for epoch in range(tcfg.get("epochs", 1)):
        if sampler:
            sampler.set_epoch(epoch)
        for batch in loader:
            it0 = time.time()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                state, _, _ = stream_forward(model, batch, device)
                occ_prob, _, sem_logit = splat_occupancy(state, grid)
                loss = official_occupancy_loss(
                    occ_prob, sem_logit, batch["label"].to(device),
                    batch["mask"].to(device), class_weights=class_weights,
                    ce_weight=lcfg.get("ce_weight", 10.0),
                    lovasz_weight=lcfg.get("lovasz_weight", 1.0),
                    free_class=FREE_CLASS)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               tcfg.get("grad_max_norm", 35.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step > warmup_iters:            # exclude compile/cudnn warmup
                t_timed += time.time() - it0
                timed_iters += 1
            if rank == 0 and step % tcfg.get("log_every", 20) == 0:
                print(f"epoch {epoch} step {step} "
                      f"loss {sum(losses[-20:]) / len(losses[-20:]):.4f}", flush=True)
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    if rank == 0:
        scores = evaluate(model, val_loader, grid, device, cfg,
                          max_batches=tcfg.get("eval_batches", 50))
        sec_per_iter = t_timed / max(timed_iters, 1)
        eff_batch = tcfg.get("micro_batch_per_gpu", 1) * world * accum
        steps_per_epoch = len(train_ds) // max(
            tcfg.get("micro_batch_per_gpu", 1) * world, 1)
        budget = {
            "sec_per_iter_measured": round(sec_per_iter, 3),
            "timed_iters": timed_iters,
            "effective_batch": eff_batch,
            "steps_per_epoch": steps_per_epoch,
            "hours_per_epoch_projected": round(sec_per_iter * steps_per_epoch / 3600, 2),
            "gpu_days_for_cfg_epochs": round(
                sec_per_iter * steps_per_epoch * tcfg.get("epochs", 1)
                * world / 86400, 2),
        }
        report = {"run": cfg["run_name"], "steps": step,
                  "final_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1),
                  "val": scores, "budget": budget,
                  "warmstart": (ws_report or {"mode": "none"}),
                  "wall_config": args.config}
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        curves = out_dir / "curves"
        curves.mkdir(exist_ok=True)
        (curves / "loss.json").write_text(json.dumps(losses))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 3))
            plt.plot(losses)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.tight_layout()
            plt.savefig(curves / "loss.png", dpi=120)
        except ImportError:
            pass                                 # loss.json is the record
        torch.save({"model": encoder.state_dict(), "config": cfg},
                   out_dir / "ckpt_last.pt")
        print(json.dumps(report, indent=2))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
