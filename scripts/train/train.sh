#!/usr/bin/env bash
# 本地单机训练启动脚本：以 dspark Qwen3-4B 为例拉起 train.py，多卡由脚本内 spawn 管理。

# Local launch mirrors the repo's node launcher, not standard
# torchrun semantics. train.py spawns one worker per visible GPU by itself.
# init_dist defaults to single-node local run when MASTER_ADDR/MASTER_PORT and
# RANK/WORLD_SIZE are unset. Total GPU workers come from CUDA_VISIBLE_DEVICES.

# Available public configs:
## dflash
#   config/dflash/dflash_gemma4_12b.py
#   config/dflash/dflash_qwen3_4b.py
#   config/dflash/dflash_qwen3_8b.py
#   config/dflash/dflash_qwen3_14b.py
## dspark
#   config/dspark/dspark_gemma4_12b.py
#   config/dspark/dspark_qwen3_4b.py
#   config/dspark/dspark_qwen3_8b.py
#   config/dspark/dspark_qwen3_14b.py
## eagle3
#   config/eagle3/eagle3_gemma4_12b.py
#   config/eagle3/eagle3_qwen3_4b.py
#   config/eagle3/eagle3_qwen3_8b.py
#   config/eagle3/eagle3_qwen3_14b.py

target_cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/qwen3_4b_target_cache}  # 目标缓存目录：可用同名环境变量覆盖，默认取 prepare_data.sh Step 3 的输出路径

# --opts overrides any config field by dotted key path: --opts "<key.path>=<value>".
# Values are parsed as Python scalars (int/float/bool/str). Repeat the flag to set
# multiple fields, e.g.:
#   --opts "data.target_cache_path=${target_cache_dir}" \
#   --opts "train.lr=3e-4" \
#   --opts "train.local_batch_size=2"
#
# local_batch_size is the per-GPU micro-batch size. Raise it to better utilize GPUs
# with more memory (e.g. 4 or 8 on 80GB cards), or keep it at 1 if you hit OOM.
# Override it without editing the config via:
#   --opts "train.local_batch_size=4"
# CUDA_VISIBLE_DEVICES 决定 worker 数（每张可见卡一个）；--opts 在不改动 config 文件的前提下覆盖字段，
# 值按 Python 字面量解析（int/float/bool/str）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --opts "data.target_cache_path=${target_cache_dir}"
