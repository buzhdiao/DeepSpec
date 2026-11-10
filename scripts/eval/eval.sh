# Local launch mirrors the repo's node launcher, not standard torchrun
# semantics. eval.py spawns one worker per visible GPU by itself.
# init_dist defaults to single-node local run when MASTER_ADDR/MASTER_PORT and
# RANK/WORLD_SIZE are unset. Total GPU workers come from CUDA_VISIBLE_DEVICES.

# Match this to the target model used by the draft checkpoint.
# 目标模型路径：必须与 draft checkpoint 训练时所用 target model 一致（否则隐状态维度/层号不匹配）。
target_name_or_path=Qwen/Qwen3-4B

# Training writes checkpoints under ~/checkpoints/<project_name>/<exp_name>/step_*.
# Use step_latest for the most recent checkpoint, or replace it with step_<N>.
# 待评测 draft model checkpoint 路径：step_latest 指向最近一次保存，也可换成具体 step_<N>。
draft_name_or_path=${HOME}/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
# 拉起评测：eval.py 会对 CUDA_VISIBLE_DEVICES 里的每张卡各 spawn 一个评测 worker；
# 全部 9 个默认任务（gsm8k/math500/aime25/humaneval/mbpp/livecodebench/mt-bench/alpaca/arena-hard-v2）
# 按 eval.py 中 TASKS 顺序执行，无需在命令行指定
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path ${target_name_or_path} \
    --draft_name_or_path ${draft_name_or_path}
