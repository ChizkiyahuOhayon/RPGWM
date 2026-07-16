"""Stage-A trainer CPU smoke + image-pipeline geometry + official loss recipe."""
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]


def test_image_pipeline_geometry():
    """Scale+crop must move a projected pixel exactly like the image resize:
    u' = s*u, v' = s*v - crop_top; and 900x1600 -> 256x704 means s=0.44,
    crop_top=140 (BEVDet-style top crop), never a direct aspect-breaking
    resize."""
    from rpgwm.data.images import ImagePipeline
    pipe = ImagePipeline(raw_hw=(900, 1600), final_hw=(256, 704))
    assert abs(pipe.scale - 0.44) < 1e-9
    assert pipe.crop_top == 140

    ego2img = torch.eye(4)
    ego2img[0, 0] = ego2img[1, 1] = 1000.0   # fx, fy
    ego2img[0, 2], ego2img[1, 2] = 800.0, 450.0
    adj = pipe.apply_to_projection(ego2img)
    pt = torch.tensor([0.3, -0.2, 5.0, 1.0])

    raw = ego2img @ pt
    u, v = (raw[0] / raw[2]).item(), (raw[1] / raw[2]).item()
    new = adj @ pt
    u2, v2 = (new[0] / new[2]).item(), (new[1] / new[2]).item()
    assert abs(u2 - (u * pipe.scale)) < 1e-4
    assert abs(v2 - (v * pipe.scale - pipe.crop_top)) < 1e-4


def test_official_loss_recipe():
    from rpgwm.losses import official_occupancy_loss
    torch.manual_seed(0)
    B, V, C = 1, 96, 17
    gt = torch.randint(0, C + 1, (B, V))
    visible = torch.rand(B, V) < 0.9
    gt_occ = (gt != C).float()
    sem_perfect = F.one_hot(gt.clamp_max(C - 1), C).float() * 10.0

    good = official_occupancy_loss(gt_occ * 0.98 + 0.01, sem_perfect, gt,
                                   visible, free_class=C)
    bad = official_occupancy_loss(1.0 - (gt_occ * 0.98 + 0.01), sem_perfect,
                                  gt, visible, free_class=C)
    assert good < bad
    # per-class weights change the loss (weighted CE really wired through)
    w = torch.ones(C + 1)
    w[C] = 0.5
    weighted = official_occupancy_loss(gt_occ * 0.7 + 0.1, sem_perfect, gt,
                                       visible, class_weights=w, free_class=C)
    unweighted = official_occupancy_loss(gt_occ * 0.7 + 0.1, sem_perfect, gt,
                                         visible, free_class=C)
    assert not torch.isclose(weighted, unweighted)
    # gradients flow
    occ = (gt_occ * 0.6 + 0.2).requires_grad_(True)
    sem = sem_perfect.clone().requires_grad_(True)
    official_occupancy_loss(occ, sem, gt, visible, free_class=C).backward()
    assert occ.grad.abs().sum() > 0 and sem.grad.abs().sum() > 0


def test_train_encoder_smoke_end_to_end():
    """Full stage-A loop on CPU: synthetic images -> streaming encoder ->
    splat -> official loss -> optimizer -> eval -> report.json with the
    measured iter/s budget block (§3.2 re-estimation input)."""
    r = subprocess.run(
        [sys.executable, "scripts/train_encoder.py",
         "--config", "configs/stage_a_smoke_cpu.yaml", "--max-steps", "3"],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    report = json.loads((REPO / "outputs/stage_a_smoke_cpu/report.json").read_text())
    assert report["steps"] == 3
    assert "miou" in report["val"]
    b = report["budget"]
    assert b["sec_per_iter_measured"] > 0 and b["timed_iters"] >= 1
    assert b["hours_per_epoch_projected"] >= 0
    assert (REPO / "outputs/stage_a_smoke_cpu/curves/loss.json").exists()
    assert (REPO / "outputs/stage_a_smoke_cpu/ckpt_last.pt").exists()


def test_train_encoder_preflight_fails_fast():
    """Missing data paths must die in seconds with the cause (60 s rule)."""
    import yaml
    cfg = yaml.safe_load((REPO / "configs/stage_a.yaml").read_text())
    cfg["data"]["index_train"] = "/nonexistent/index.json"
    bad = REPO / "outputs/_preflight_bad.yaml"
    bad.parent.mkdir(exist_ok=True)
    bad.write_text(yaml.safe_dump(cfg))
    r = subprocess.run(
        [sys.executable, "scripts/train_encoder.py", "--config", str(bad)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 2
    assert "PREFLIGHT FAILED" in r.stderr and "index_train" in r.stderr
