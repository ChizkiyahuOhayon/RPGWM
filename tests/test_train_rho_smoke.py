import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_train_rho_smoke_end_to_end():
    """Full M2 loop on CPU: stage-B smoke ckpt -> pass1 error stats + quantile
    fit -> pass2 rho-head training -> val ECE -> artifacts."""
    ckpt = REPO / "outputs/smoke_cpu/ckpt_last.pt"
    if not ckpt.exists():  # produce the stage-B checkpoint first
        r0 = subprocess.run([sys.executable, "scripts/train.py", "--config",
                             "configs/smoke_cpu.yaml", "--max-steps", "2"],
                            cwd=REPO, capture_output=True, text=True, timeout=600)
        assert r0.returncode == 0, r0.stderr

    r = subprocess.run(
        [sys.executable, "scripts/train_rho.py", "--config", "configs/smoke_cpu.yaml",
         "--ckpt", str(ckpt), "--max-batches", "2", "--epochs", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    out = REPO / "outputs/smoke_cpu_rho"
    report = json.loads((out / "report.json").read_text())
    assert report["pass1_valid_errors"] > 0
    assert len(report["single_step_pos_var"]) == 3
    assert (out / "rho_head.pt").exists() and (out / "quantiles.pt").exists()
