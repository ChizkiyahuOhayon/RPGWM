import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_train_smoke_end_to_end(tmp_path):
    """The full stage-B loop (synthetic data -> rollout -> splat -> loss ->
    optimizer -> eval vs copy-last-frame -> report.json) must run on CPU."""
    r = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", "configs/smoke_cpu.yaml",
         "--max-steps", "2"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    report = json.loads((REPO / "outputs/smoke_cpu/report.json").read_text())
    assert report["steps"] == 2
    assert "model" in report["scores"] and "baseline" in report["scores"]
    assert (REPO / "outputs/smoke_cpu/ckpt_last.pt").exists()
