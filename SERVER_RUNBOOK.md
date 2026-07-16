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

## 1.5 A-G 教师流水线启动（最高优先，W1 第一天就挂上；GPU 1，与其余步骤并行）

目的：Vista/DriveDreamer-2 权重落盘 + 单 clip 抽取冒烟 + 吞吐实测，锁定 A-G
分辨率与范围。Gate-1.5 在 W6，下载 + 90–170GB 缓存是全项目最长前置链，一天都不等。

```bash
# 独立 vista 环境（与 mm-stack / navsim 隔离，见环境三分表）
conda create -n vista python=3.9 -y && conda activate vista
cd third_party/Vista && pip install -r requirements.txt   # torch>=2.0.1+xformers
# 权重（国内走 hf-mirror）：
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download OpenDriveLab/Vista vista.safetensors --local-dir ../../ckpts/
# DriveDreamer-2 权重一并落盘存档（消融 A-G2 + 教师备选链），路径登记 configs/paths.yaml

# 冒烟三连（顺序执行,每步失败都会 60s 内非零退出并报因）：
cd ../..
CUDA_VISIBLE_DEVICES=1 python scripts/ag_probe_vista.py --ckpt ckpts/vista.safetensors --list-blocks
CUDA_VISIBLE_DEVICES=1 python scripts/ag_probe_vista.py --ckpt ckpts/vista.safetensors \
    --t-star 1 --blocks <上一步选定的 2-3 个 output_blocks 名> 
cat outputs/ag_probe/report.json
```
- 验收：report.json 给出 (a) 各 block 特征形状；(b) 秒/clip；(c) MB/clip@fp16 与全量
  缓存投影——落在 §3.1 预算（3–6MB/clip、90–170GB、A-G≈12 GPU-days）内则锁定
  分辨率/尺度并记录进 configs/paths.yaml；超预算先降分辨率再报告。
- 注意：probe 是按 Vista @cc9821b 写的、未在本地跑过（CPU 盒无权重无 CUDA）；
  UNet 调用签名若有漂移，脚本会指出需要适配的行号——这属于 W1 预期工作量。

## 1.6 spconv 验证（5 分钟，rpgwm 环境）

```bash
conda activate rpgwm && pip install spconv-cu120   # torch 2.4/cu121 下通常兼容
python -c "
import torch, spconv.pytorch as spconv
from rpgwm.models.encoder import GaussianEncoder
e = GaussianEncoder(num_slots=64, embed_dims=32, num_classes=6, feat_dim=16,
                    num_blocks=1, num_cams=2, self_interact='spconv').cuda()
print('spconv path OK')"
```
- 过 → 阶段 A 用 `self_interact: spconv`（热启 op 覆盖率 100%，kNN 版降级 fallback）；
- 不过（版本冲突等）→ 如实记录报错，用 kNN fallback（零初始化+残差位 = 热启无损）。

## 2. Baseline 复现之一：GaussianFormer-2 推理（≈0.5 GPU-day）

目的：验证数据管线 + 拿到编码器热启权重。**只推理，不重训。**
（submodule 已随主仓库带上，服务器上 `git submodule update --init` 即可。）

```bash
cd third_party/GaussianFormer && pip install -r requirements.txt
# 按其 docs/installation.md 装 mm-stack + 4 个 CUDA ops（含 GF-2 的 localagg_prob*）
# ckpt：README 表中 Prob-64（config/prob/nuscenes_gs6400.py 对应权重）
CUDA_VISIBLE_DEVICES=0 python eval.py --py-config config/prob/nuscenes_gs6400.py \
    --work-dir out/gf2_repro --resume-from ckpts/gaussianformer2_prob64.pth
```
- **本步定位（2026-07-16 重定义）：环境自检,不是热启证据。** 我们的编码器
  架构 ≠ 官方（R50@256×704 vs R101-DCN@1600×864），复现不出我方数字是正常的。
  1. **环境自检**：官方仓库原样（官方 config + 官方 localagg_prob CUDA 核）复现
     Prob-64 的 SurroundOcc mIoU 20.04（README:111），差 <0.3 → 只证明环境与
     数据没问题；
  2. **热启报告存档**：`load_gf2_partial` 的 block_coverage（分母只含 Gaussian
     block；backbone/FPN/lifter 在 NOT_TRANSFERRED 清单里，理由见
     rpgwm/models/gf2_warmstart.py 头注释）。此数字不是 gate；
  3. **热启唯一验收 = 阶段 A 的 1/4-split A/B**：`configs/stage_a.yaml` 分别以
     `warmstart: gf2` / `warmstart: none` 各跑 1–2 epoch，比 report.json 的
     val miou，谁赢用谁，结果记录在案；
  4. **类序断言**：用 `data/occ3d/annotations.json` 的 category 列表核对
     `rpgwm/models/gf2_warmstart.py` 里 OCC3D_CLASS_NAMES（顺序必须逐项一致，
     不一致改代码里的表并重跑 test_semantic_alignment_guard）。
- **两个 gate 不许混**：上面 1 是编码器环境自检;我们的可微 splat 对拍官方
  localagg_prob 核（同一组 Gaussian 输入,占据概率差 <1e-4）是**另一条**
  独立验收（splat 保真度,服务 Gate-1),完成后各自记录。
- 产物：权重路径 + 各项验收记录登记进 `configs/paths.yaml`。

## 2.5 阶段 A 训练（W2–3，GPU 0；A-G 在 GPU 1 上继续）

```bash
# 先冒烟（合成数据已在本地过,这里用真实 mini 索引再冒一次烟）
python scripts/build_index.py --nuscenes-root data/nuscenes --version v1.0-mini \
    --split train --out data/index_mini_train.json     # 含 cams 记录
CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config configs/stage_a.yaml \
    --max-steps 50        # 短跑：确认 loss 下降 + report.json 的 budget 块有数
# report.json -> budget.sec_per_iter_measured / hours_per_epoch_projected /
# gpu_days_for_cfg_epochs —— 把这三个数报回来,用于重估 §3.2 阶段 A 行
```
- A/B 两臂（warmstart gf2/none）在 1/4-split 上排队跑,不抢卡；
- class_weights 全量训练前按 Occ3D train 频率重算（GF-2 的表是 SurroundOcc 拟合）；
- **收敛监察**：val-mIoU 曲线 W3 末未走平立即上报,不等 W4 Gate-1。

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
pip install nuscenes-devkit pyquaternion   # 仅本步需要

# 4a. 建索引（devkit 只在这一步用到）
python scripts/build_index.py --nuscenes-root data/nuscenes --version v1.0-mini \
    --split train --out data/index_mini_train.json
python scripts/build_index.py --nuscenes-root data/nuscenes --version v1.0-mini \
    --split val --out data/index_mini_val.json

# 4b. 先用 random 编码器打通全管线（不出数字，只验证 I/O 与吞吐）
python scripts/dump_gaussians.py --index data/index_mini_train.json \
    --out data/gaussian_cache --encoder random --n 6400
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/gate1_mini.yaml --max-steps 20

# 4c. 接入 GF-2 编码器（填 scripts/dump_gaussians.py 里的 build_gf2_encoder TODO，
#     用步骤 2 验收过的 ckpt），重新 dump 缓存，然后正式训练：
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/gate1_mini.yaml
```
产物 `outputs/gate1_mini/report.json` 内含 `gate1_beats_copy_last_frame` 布尔位。
- Gate-1 验收：nuScenes-mini 上 6 步 rollout 的 forecast 显著优于 copy-last-frame 基线；
  失败 → 一周排障（先查 splatting 数值、再查动作条件化），再失败 → 启动 Plan B（见计划 §7）。

## 5. 通用纪律
- 每个实验 = 一个 config + 一条命令；结果自动落 `outputs/<run>/report.json` + git hash；
- 两卡跑主训练时，消融串行排队，不抢卡；
- 每周五把 `outputs/*/report.json` 同步回本地仓库。
