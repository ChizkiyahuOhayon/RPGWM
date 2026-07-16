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
- **验收（输出级复现，2026-07-16 起取代"覆盖率%"）**：
  1. 官方设定下 SurroundOcc mIoU 复现 Prob-64 报告值 20.04，差 <0.3；
  2. 我方热启：`load_gf2_partial` 报告存档（coverage / anchor_kept_in_range_frac /
     resampled 数），随后**未训练**的我方编码器在我们设定（R50@256×704/Occ3D）下
     跑推理评测，记录零训练 mIoU 基线值——这是阶段 A 收敛曲线的起点参照，
     伪 checkpoint 的 100% 只证明映射器自洽，不算证据；
  3. 图像预处理与官方逐项对齐后才算完成：GF-2 用 ImageNet 均值方差
     mean=[123.675,116.28,103.53] std=[58.395,57.12,57.375] to_rgb=True
     （config/_base_/surroundocc.py:8）——我方 dataloader 必须一致（/255 后即
     torchvision 标准归一化），resize 到 256×704 无 crop，核对写进对拍记录。
- 产物：权重路径 + 三项验收记录登记进 `configs/paths.yaml`。

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
