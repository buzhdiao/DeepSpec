import argparse  # 命令行参数解析：--config 与 --opts
import json  # 把最终生效的 config 序列化为 JSON 打印
import os  # 设置环境变量、探测 .git 目录
import torch  # 多进程 spawn、CUDA 设备数、fp32 矩阵乘精度
from deepspec.utils import (  # 训练入口通用工具集
    CustomJSONEncoder,  # 可序列化函数/类/Path/ConfigNode 的 JSON 编码器
    get_git_diff,  # 读取工作区相对 HEAD 的 git diff（记录实验代码状态）
    load_config,  # 以模块方式加载 config .py 文件为 ConfigNode
    parse_opts_to_config,  # 把 --opts "key.path=value" 按点路径覆盖进 config
    seed_all,  # 统一设置 python/numpy/torch 随机种子
    get_git_sha,  # 读取当前 commit SHA（可附带远端/分支详情）
)

os.environ['USE_TORCH']='true'  # 环境开关：标记 torch 后端运行环境（沿用上游训练框架约定）
os.environ['WANDB_DISABLED']='true'  # 禁用 wandb 在线上报，训练日志只走本地
os.environ['TOKENIZERS_PARALLELISM']='false'  # 关闭 HF tokenizer 多线程，避免 spawn 多进程下的 fork 告警/死锁
torch.set_float32_matmul_precision("high")  # fp32 matmul 允许 TF32：A100/H100 上大幅提速，精度损失可忽略


def parse_args():
    # 解析命令行并合成最终 config：先加载 config 文件，再用 --opts 按点路径逐项覆盖。
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)  # config .py 路径（如 config/dspark/dspark_qwen3_4b.py）
    parser.add_argument("--opts", action="append", default=[])  # 可重复出现的字段覆盖项，值按 Python 字面量解析
    args = parser.parse_args()  # 原始参数 Namespace
    config = parse_opts_to_config(args.opts, load_config(args.config))  # 合并 CLI 覆盖，并执行 config 内可选的 finalize_cfg 钩子
    config._origin_config_path = os.path.abspath(args.config)  # 记录原始 config 绝对路径：随 checkpoint 落盘，便于复现实验
    config._origin_opts = list(args.opts)  # 记录原始 --opts 列表，与 config 路径配套用于复现
    return config


def main(local_rank):
    # 单个 GPU worker 的训练主流程；local_rank 由 torch.multiprocessing.spawn 按进程序号注入。
    args = parse_args()  # 每个 worker 各自解析一遍 config（互不共享，进程隔离）
    seed_all(int(args.seed))  # 固定随机种子：数据 shuffle、初始化等在各 rank 一致可复现
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)  # 本地 0 号卡打印最终生效的完整 config
    trainer = args.train.trainer_cls(local_rank, args)  # config 指定 trainer 类（如 Qwen3DSparkTrainer）；构造时完成 init_dist、加载 target cache、建 FSDP 模型
    trainer.train()  # 执行训练循环：训练 draft model、定期保存 checkpoint
    trainer.clean_up()  # 释放模型/数据集/进程组等资源


if __name__ == "__main__":
    if os.path.exists(".git"):  # 仅在仓库内运行时打印 git 信息（外部部署目录无 .git）
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))  # 打印 commit SHA 及远端/分支，标记实验代码版本
        print("git diff:", get_git_diff())  # 打印未提交改动，保证实验可复现
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())  # 每张可见 GPU 拉起一个 main 子进程；单机多卡不走 torchrun
