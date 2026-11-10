#!/usr/bin/env bash
# Qwen3-4B 训练数据一键准备：下载并切分 PerfectBlend → 用 sglang server 重生成回复 → 导出目标隐状态缓存。
# 对应三个脚本：download_and_split.py / generate_train_data.py / prepare_target_cache.py。
set -euo pipefail  # 严格模式：未定义变量报错、任一命令失败即退出、管道中段失败也视为失败

model_path=Qwen/Qwen3-4B  # 生成/导出缓存所用的目标模型（HF 名称）
config_path=config/dspark/dspark_qwen3_4b.py  # Step 3 导出缓存读取的训练 config（定层号/模板/长度）

dataset_name=mlabonne/open-perfectblend  # 源数据集：PerfectBlend 对话混合集
test_size=0.05  # 切给评测集的比例（5%，对齐 SpecForge 默认）
train_split_path=train_datasets/perfectblend_train.jsonl  # Step 1 输出：原始切分的训练对话 JSONL
eval_data_dir=eval_datasets  # Step 1 输出：评测 turns 文件目录（生成 perfectblend.jsonl）

train_data_path=train_datasets/qwen3_4b/perfectblend_train_regen.jsonl  # Step 2 输出：重生成回复后的训练数据
cache_dir=${HOME}/.cache/deepspec/qwen3_4b_target_cache  # Step 3 输出：目标模型隐状态缓存目录

server_host=127.0.0.1  # sglang server 主机地址（单机部署）
num_workers=8  # sglang server 实例数，须与 launch_sglang_server.sh 的 num_workers 一致
start_port=30000  # sglang server 起始 HTTP 端口，同上须一致
concurrency=32  # 每 server 的并发请求上限
temperature=0.7  # 重生成采样温度
top_p=0.8  # nucleus 采样截断
top_k=20  # top-k 采样
min_p=0  # min-p 采样（0 等于不启用）
max_tokens=4096  # 单次生成最大 token 数

server_addresses=()  # 汇总全部 server 地址
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do  # 按 实例序号 生成 host:port 列表
    server_addresses+=("${server_host}:$((start_port + worker_id))")  # 第 i 台监听 start_port+i
done

echo "Step 1/3: downloading and splitting ${dataset_name}"
# 下载 PerfectBlend 并按比例切分；--skip-existing 保证重跑时已有输出则直接跳过
python scripts/data/download_and_split.py \
    --dataset-name "${dataset_name}" \
    --test-size "${test_size}" \
    --train-output-path "${train_split_path}" \
    --test-output-dir "${eval_data_dir}" \
    --skip-existing

mkdir -p "$(dirname "${train_data_path}")"  # 确保 Step 2 输出文件的父目录存在

echo "Step 2/3: generating Qwen3-4B train data: ${train_data_path}"
echo "Start sglang first with: bash scripts/data/launch_sglang_server.sh"  # 提示须先另行启动 sglang server
# 对 Step 1 的训练对话逐轮重新生成 assistant 回复（替换原始 gpt 回复），作为 draft model 的训练目标；
# --resume 支持断点续跑；采样参数与 launch_sglang_server.sh 部署的模型配套
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_addresses[@]}" \
    --concurrency "${concurrency}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --min-p "${min_p}" \
    --max-tokens "${max_tokens}" \
    --disable-thinking \
    --resume \
    --input-file-path "${train_split_path}" \
    --output-file-path "${train_data_path}"

echo "Stop sglang before Step 3 if it is using the same GPUs."  # Step 3 需占满 GPU 前向，先释放 sglang
echo "Step 3/3: preparing Qwen3-4B target cache: ${cache_dir}"
# 用目标模型对重生成后的训练数据做前向，导出目标层隐状态缓存（多卡并行，每卡一个 worker）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size 16
