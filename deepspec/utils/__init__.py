# deepspec.utils 统一出口：聚合配置加载、分布式协作、指标统计、优化器、IO 等训练工具，
# 训练器与评测器统一通过 `from deepspec.utils import ...` 取用。
import random  # Python 内置随机数：与 torch/np 种子一起保证复现

import numpy as np  # NumPy：随机种子与数组操作
import torch  # PyTorch：张量操作、CUDA 设备属性与随机种子

from .config import CustomJSONEncoder, jsonable, load_config, parse_opts_to_config  # 配置加载/命令行覆盖与 JSON 序列化
from .distributed import (  # 分布式初始化、主进程判断与无状态续训采样器
    StatelessResumableDistributedSampler,  # 可断点续训的确定性分布式采样器
    init_dist,  # 初始化 NCCL 进程组并返回设备与全局 rank/world_size
    is_global_main_process,  # 是否为全局 rank 0
    is_local_main_process,  # 是否为本机 0 号卡
    main_process_first,  # 主进程先执行的屏障上下文管理器
    print_on_global_main,  # 仅全局 rank 0 打印（带时间戳）
    print_on_local_main,  # 仅本机 0 号卡打印（带时间戳）
)
from .io import ensure_dir, safe_symlink  # 递归建目录与原子符号链接更新
from .metrics import add_metric, flush, reset  # 指标记录、日志窗口汇总与清零
from .optim import BF16Optimizer  # bf16 混合精度优化器（fp32 主参数 + warmup 余弦调度）

# 统一播种所有随机数发生器，保证训练完全可复现：torch（CPU/GPU）、Python random、
# NumPy 三处独立维护状态，漏掉任何一处都会破坏确定性。
def seed_all(seed):  # seed：全局随机种子
    torch.manual_seed(seed)  # CPU 端 torch 随机数
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy 随机数（数据增强等）
    torch.cuda.manual_seed_all(seed)  # 所有可见 GPU 的随机数（多卡必须，单卡也覆盖）

# 获取当前代码的 git 短 SHA；detail_info=True 时额外返回 git status 与最近一次提交，
# 用于把"运行时代码目录状态"写进训练日志，便于事后追溯实验对应的确切代码版本。
def get_git_sha(detail_info=False):  # detail_info：是否附带详细 git 信息
    import subprocess  # 延迟导入：仅用到 git 信息时才加载子进程模块

    # 执行 git 命令并以 UTF-8 文本返回；在非 git 目录（如线上容器）中静默降级为 "unknown"。
    def run_git_text(cmd):  # cmd：git 命令及参数列表
        try:
            return subprocess.check_output(  # 运行命令并捕获标准输出
                cmd,  # 待执行的命令
                encoding="utf-8",  # 直接解码为 str 而非返回 bytes
                errors="replace",  # 无法解码的字节替换为 U+FFFD，避免脏输出抛异常
            ).rstrip()  # 去掉末尾换行
        except Exception:  # 非 git 仓库、git 不存在等任何失败都不影响训练
            return "unknown"  # 降级占位值

    cmd = ["git", "rev-parse", "--short", "HEAD"]  # 取当前提交的短 SHA
    sha = run_git_text(cmd)  # 形如 "1a2b3c4" 的短哈希
    if not detail_info:  # 默认只要 SHA
        return sha
    # 获取一些 git 详细信息，帮助从训练日志中得到运行时的目录状态
    # 查看当前目录下没有提交的文件信息
    cmd = ["git", "status"]  # 未提交改动：判断运行目录是否干净
    status = run_git_text(cmd)
    # 查看当前目录下上一次提交的信息
    cmd = ["git", "log", "-n", "1"]  # 最近一次提交的完整说明
    last_commit = run_git_text(cmd)
    return sha, status, last_commit  # (短 SHA, git status 输出, 最近提交信息)


# 返回工作区与 rev（默认 HEAD）的 diff 文本，可传自定义 ref 对比断点续训前后的
# 代码改动；失败时返回 "unknown"，不阻断训练。
def get_git_diff(rev="HEAD"):  # rev：对比基准（提交号/分支名）
    import subprocess  # 延迟导入子进程模块

    try:
        return subprocess.check_output(  # 运行 git diff 并捕获输出
            ["git", "diff", rev],  # 工作区相对 rev 的差异
            encoding="utf-8",  # 解码为 str
            errors="replace",  # 解码失败用替换字符兜底
        ).rstrip()  # 去掉末尾换行
    except Exception:  # 非 git 目录等情况
        return "unknown"  # 降级占位值


# 对外导出的公共 API 白名单：`from deepspec.utils import *` 时只暴露以下名字。
__all__ = [
    "BF16Optimizer",  # bf16 混合精度优化器
    "CustomJSONEncoder",  # 支持 config 常见类型的 JSON 编码器
    "StatelessResumableDistributedSampler",  # 无状态可续训分布式采样器
    "add_metric",  # 记录一条标量/比率指标
    "ensure_dir",  # 递归创建目录
    "flush",  # 汇总并清空本日志窗口的指标
    "get_git_diff",  # 读取 git diff
    "get_git_sha",  # 读取 git 短 SHA 与详细信息
    "init_dist",  # 初始化分布式进程组
    "is_global_main_process",  # 是否全局主进程
    "is_local_main_process",  # 是否本机主进程
    "jsonable",  # 递归转为可 JSON 序列化结构
    "load_config",  # 从 .py 文件加载配置
    "main_process_first",  # 主进程先行的上下文管理器
    "parse_opts_to_config",  # 应用 --opts 命令行覆盖
    "print_on_global_main",  # 全局主进程打印
    "print_on_local_main",  # 本机主进程打印
    "reset",  # 清空指标缓存
    "safe_symlink",  # 原子更新符号链接
    "seed_all",  # 统一设置随机种子
]
