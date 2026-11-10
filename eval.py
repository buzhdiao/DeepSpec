from __future__ import annotations  # 延迟注解求值：允许注解使用 X | None 等新语法
import argparse  # 命令行参数解析
import json  # 打印参数时序列化为 JSON
import torch  # 多进程 spawn、CUDA 设备数
from transformers import AutoConfig  # 读取 draft model config，取 architectures 做评测器路由
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator  # dspark（confidence-head 风格）评测器
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator  # eagle3 风格评测器
from deepspec.utils import CustomJSONEncoder  # 可序列化函数/类/Path/ConfigNode 的 JSON 编码器

EVALUATORS = {  # draft config 的 architectures[0] → 评测器类 的路由表
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,  # Qwen3 系 dspark draft model
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,  # Gemma4 系 dspark draft model
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,  # Qwen3 系 eagle3 draft model
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,  # Gemma4 系 eagle3 draft model
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,  # 兼容旧命名的 eagle3 draft model，按 Qwen3 评测
}

TASKS = [  # 默认评测任务列表：(任务名, 样本数)。名称对应 eval_datasets/<task>.jsonl
    ("gsm8k", 500),  # 小学数学应用题（OpenAI GSM8K）
    ("math500", 500),  # MATH 竞赛数学的 500 题子集
    ("aime25",30),  # 2025 年 AIME 竞赛题（30 题）
    ("humaneval", 164),  # OpenAI Python 代码生成（164 题）
    ("mbpp", 256),  # 基础 Python 编程题（ Mostly Basic Python Problems）
    ("livecodebench", 500),  # 竞赛式代码生成基准 LiveCodeBench
    ("mt-bench", 80),  # 多轮对话质量评测（80 组双轮提问）
    ("alpaca", 500),  # 指令跟随通用问答（Alpaca 数据）
    ("arena-hard-v2", 500),  # LMArena 困难用户提问子集 v2
]

def parse_args():
    # 解析评测命令行参数；评测任务固定为上面的 TASKS 全集，不接受外部指定。
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)  # 目标模型（被加速的大模型）HF 名称或本地路径
    parser.add_argument("--draft_name_or_path",type=str,required=True)  # 待评测的 draft model checkpoint 路径
    parser.add_argument("--max-new-tokens", type=int, default=2048)  # 每条样本最大新生成 token 数
    parser.add_argument("--temperature", type=float, default=1.0)  # 采样温度（1.0 为按模型原始分布采样）
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)  # TensorBoard 日志目录；None 表示不写
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)  # 写入 TensorBoard 时标记的训练步数
    parser.add_argument("--seed", type=int, default=980406)  # 随机种子，保证评测可复现
    args = parser.parse_args()
    args.tasks = list(TASKS)  # 把默认任务列表挂到参数上（顺序执行全部 9 个任务）
    return args


def main(local_rank: int, args):
    # 单个 GPU worker 的评测主流程；local_rank 由 torch.multiprocessing.spawn 注入。
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)  # 0 号卡打印全部评测参数
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)  # 读 draft model config
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]  # 按 architectures[0] 路由到对应评测器类
    evaluator = evaluator_cls(local_rank, args)  # 构造评测器：内部加载 target/draft 模型与数据集
    evaluator.evaluate()  # 逐任务生成、打分并输出准确率等指标
    evaluator.clean_up()  # 释放模型与进程组资源

if __name__ == "__main__":
    args = parse_args()  # 主进程先解析一次参数，再整体传给各子进程
    torch.multiprocessing.spawn(  # 每张可见 GPU 拉起一个 main 子进程（单机多卡，不走 torchrun）
        main,
        args=(args,),  # 除 spawn 注入的 local_rank 外的固定参数
        nprocs=torch.cuda.device_count(),  # worker 数 = 可见 GPU 数
    )
