# 公共常量：默认模型 HF 路径、缓存/输出目录、自动评测命令等，供各 config/*.py 复用。
import os  # os.path.expanduser：把路径中的 ~ 展开为用户主目录

## cache dir
CACHE_DIR = os.path.expanduser("~/.cache/deepspec")  # 默认缓存根目录（数据集缓存等，见 data/jsonl_dataset.py）

## model_name_or_path
QWEN_3_4B = "Qwen/Qwen3-4B"  # Qwen3-4B 权重路径（HF Hub 名；本地不存在时自动下载）
QWEN_3_8B = "Qwen/Qwen3-8B"  # Qwen3-8B
QWEN_3_14B = "Qwen/Qwen3-14B"  # Qwen3-14B
GEMMA_4_12B = "google/gemma-4-12B-it"  # Gemma-4-12B 指令微调版（目标模型）
BASE_TB_DIR = os.path.expanduser("~/tensorboard")  # TensorBoard 日志根目录（finalize_cfg 拼接项目/实验名）
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")  # checkpoint 根目录（同上拼接）

## auto eval
auto_eval_command = None  # 训练完成后的自动评测命令模板；None 表示不触发自动评测