# RP-GWM · 2×A40 服务器 Runbook（W1）

服务器：2×NVIDIA A40 48GB（`CUDA_VISIBLE_DEVICES=0,1`）。本文件里的命令按顺序复制粘贴即可；每步末尾有"验收"判据，不达标不进入下一步。

## 0. 环境（一次性）

```bash
# 国内镜像（按需）
export HF_ENDPOINT=https://hf-mirror.com
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

conda create -n rpgwm python=3.10 -y && conda activate rpgwm
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
git clone <本仓库> && cd RPGWM && pip install -r requirements.txt
python -m pytest tests/ -q        # 验收：26 passed（和本地一致才继续）
```

## 1. 数据（一次性）

```
data/
├── nuscenes/            # v1.0-trainval（先下 v1.0-mini 打通流程）
│   ├── samples/  sweeps/  v1.0-trainval/  ...
└── occ3d/               # Occ3D-nuScenes（gts/ 含 labels.npz: semantics + mask_camera）
```
- nuScenes: https://www.nuscenes.org/download （mini ≈ 4GB 先行）
- Occ3D-nuScenes: https://github.com/Tsinghua-MARS-Lab/Occ3D （百度网盘/GDrive 链接在其 README）
- 验收：`python -c "import numpy as np; d=np.load('data/occ3d/gts/<任一场景>/<任一token>/labels.npz'); print(d['semantics'].shape, d['mask_camera'].shape)"` 输出 `(200,200,16) (200,200,16)`。

## 2. Baseline 复现之一：GaussianFormer-2 推理（≈0.5 GPU-day）

目的：验证数据管线 + 拿到编码器热启权重。**只推理，不重训。**

```bash
git clone https://github.com/huang-yh/GaussianFormer.git third_party/GaussianFormer
cd third_party/GaussianFormer && pip install -r requirements.txt
# 按其 README 装 CUDA 扩展（pointops / local_aggregate 等）
# 下载其 released ckpt（GaussianFormer-2 / prob 分支，nuScenes-SurroundOcc 配置）
CUDA_VISIBLE_DEVICES=0 python eval.py --py-config config/<gf2配置>.py \
    --work-dir out/gf2_repro --resume-from ckpts/<gf2权重>.pth
```
- 验收：SurroundOcc 协议 mIoU 与论文报告值（≈20.8/GF-2）差 <0.3。
- 产物：确认可用的编码器权重路径，登记进 `configs/paths.yaml`。

## 3. Baseline 复现之二：forecasting 评测协议对拍（≈0.2 GPU-day）

目的：我们的 `rpgwm/eval/forecast.py` 必须和 OccWorld 官方评测在同一输入上出同一个数。

```bash
git clone https://github.com/wzzheng/OccWorld.git third_party/OccWorld
# 用其 released ckpt 跑官方 eval 存 per-frame 预测；再用我们的 forecast_scores 重打分
CUDA_VISIBLE_DEVICES=1 python scripts/crosscheck_eval.py \
    --occworld-dump out/occworld_pred --occ3d-root data/occ3d
```
- 验收：两套评测的 1/2/3s mIoU 差 <0.05（浮点/边界差异容忍）。**不对拍不许报任何表 1 数字。**
- （可选加分）SparseWorld-TC 代码已开源：https://github.com/MrPicklesGG/SparseWorld ，其 splatting/Chamfer 监督代码可对照参考。

## 4. Gate-1 mini 训练（W2-W4，≈2-4 GPU-days）

```bash
# 阶段 A'（mini）：编码器热启 GF-2 权重，短程适配 N=6400 + 流式搬运
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/gate1_mini.yaml
```
- Gate-1 验收：nuScenes-mini 上 6 步 rollout 的 forecast 显著优于 copy-last-frame 基线；
  失败 → 一周排障（先查 splatting 数值、再查动作条件化），再失败 → 启动 Plan B（见计划 §7）。

## 5. 通用纪律
- 每个实验 = 一个 config + 一条命令；结果自动落 `outputs/<run>/report.json` + git hash；
- 两卡跑主训练时，消融串行排队，不抢卡；
- 每周五把 `outputs/*/report.json` 同步回本地仓库。
