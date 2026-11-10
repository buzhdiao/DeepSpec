"""Convert supported Hugging Face eval datasets to local turns JSONL files.

Usage:
    python eval_datasets/convert_eval_datasets_to_jsonl.py openai/gsm8k \
        --output-path /tmp/gsm8k.jsonl

By default, the converter writes to eval_datasets/ and raises an error when the
target JSONL file already exists.

Supported datasets:

| Dataset | Hugging Face dataset name |
| --- | --- |
| gsm8k | openai/gsm8k |
| math500 | HuggingFaceH4/MATH-500 |
| aime24 | HuggingFaceH4/aime_2024 |
| aime25 | MathArena/aime_2025 |
| alpaca | tatsu-lab/alpaca |
| mt-bench | HuggingFaceH4/mt_bench_prompts |
| humaneval | openai/openai_humaneval |
| mbpp | google-research-datasets/mbpp |
| lbpp | CohereLabs/lbpp |
| swe-bench | princeton-nlp/SWE-bench_Lite |
| livecodebench | livecodebench/code_generation_lite |

PerfectBlend comes from mlabonne/open-perfectblend, but the checked-in
perfectblend.jsonl was built from a regenerated qwen3-4b test cache. The raw HF
download, splitting, and conversion utility lives in
scripts/data/download_and_split.py.
"""

from __future__ import annotations  # 延迟注解求值：允许注解使用 X | None 等新语法

import argparse  # 命令行参数解析
import json  # 逐行读 JSONL 原始文件、写转换结果
from collections.abc import Callable  # 类型注解：格式化回调的签名
from dataclasses import dataclass  # 声明数据集描述结构体
from pathlib import Path  # 路径对象：输出目录/文件处理


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent  # 默认输出根目录 = 本文件所在目录（eval_datasets/）
REASONING_SUFFIX = (  # 数学类任务统一追加的思维链提示：要求分步推理并用 \boxed{} 给出最终答案
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


@dataclass(frozen=True)
class DatasetSpec:
    # 一个可转换数据集的完整描述：HF 来源信息 + 输出文件名 + 行 → turns 的格式化回调。
    # frozen=True：spec 是全局注册表中的共享常量，禁止运行期修改。
    output_name: str  # 输出文件名（不含 .jsonl 后缀），同时是评测任务名（如 gsm8k）
    hf_name: str  # Hugging Face 数据集仓库名（如 openai/gsm8k）
    config_name: str | None  # HF 数据集 config/子集名；None 表示该数据集无 config
    split: str  # 使用的 HF split（如 test/train）
    format_turns: Callable[[dict], list[str]]  # 一行原始数据 → turns（字符串列表，每项一轮对话）的回调
    jsonl_files: tuple[str, ...] = ()  # 非空时：不走 load_dataset，直接从 HF 仓库逐个下载这些 JSONL 原始文件
    parquet_files: tuple[str, ...] = ()  # 非空时：直接从 HF 仓库下载这些 parquet 文件解析


def format_gsm8k(row: dict) -> list[str]:
    # gsm8k 单轮：question + 统一思维链提示。
    return [f"{row['question']}{REASONING_SUFFIX}"]


def format_math_problem(row: dict) -> list[str]:
    # math500/aime24/aime25 共用：题目字段是 problem，同样追加思维链提示。
    return [f"{row['problem']}{REASONING_SUFFIX}"]


def format_alpaca(row: dict) -> list[str]:
    # alpaca 单轮：instruction 可带可选 input 字段（附加上下文），拼成一条 user 消息。
    instruction = row["instruction"]  # 指令主体
    input_text = row["input"]  # 可选附加上下文，常为空串
    if input_text:
        return [f"{instruction}\n\nInput:\n{input_text}"]  # 有附加输入：拼接到指令后
    return [instruction]  # 无附加输入：仅指令本身


def format_mt_bench(row: dict) -> list[str]:
    # mt-bench 多轮：prompt 字段本身就是多轮 user 消息列表，校验类型后原样作为 turns。
    turns = row["prompt"]  # 每轮一条 user 提问的列表
    if not isinstance(turns, list):
        raise TypeError("mt-bench field `prompt` must be a list of turns.")
    return turns


def format_humaneval(row: dict) -> list[str]:
    # humaneval 单轮：把函数签名/docstring 包进 python 代码块，要求写出能通过测试的实现。
    return [
        (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{row['prompt']}\n```"
        )
    ]


def format_mbpp(row: dict) -> list[str]:
    # mbpp 单轮：prompt 字段即完整任务描述。
    return [row["prompt"]]


def format_lbpp(row: dict) -> list[str]:
    # lbpp 单轮：任务描述在 instruction 字段。
    return [row["instruction"]]


def format_swe_bench(row: dict) -> list[str]:
    # swe-bench 单轮：给出 issue 描述（problem_statement），要求修复该问题。
    return [
        (
            f"Problem Statement:\n{row['problem_statement']}\nPlease fix the issue "
            "described above."
        )
    ]


def format_livecodebench(row: dict) -> list[str]:
    # livecodebench 两种行格式：消息列表版取全部 user 轮；平铺字段版拼标准代码生成 prompt。
    if "messages" in row:  # 消息列表版（v1 风格）
        turns = [
            message["content"]  # 只保留 user 消息内容
            for message in row["messages"]
            if message["role"] == "user"
        ]
        if not turns:
            raise ValueError("livecodebench row has no user messages.")
        return turns

    question = row["question_content"]  # 题目描述
    starter_code = row["starter_code"]  # 给定的函数签名/起始代码，可能为空
    code_template = starter_code if starter_code else "# YOUR CODE HERE"  # 代码块占位内容
    format_label = (  # 按是否有起始代码选择措辞
        "Use the following code structure:"
        if starter_code
        else "Write your code in the following format:"
    )
    return [
        (
            "You are an expert Python programmer. You will be given a question "
            "(problem specification) and will generate a correct Python program "
            "that matches the specification and passes all tests. You will NOT "
            "return anything except for the program\n\n"
            f"### Question:\n{question}\n\n"
            f"### Format: {format_label}\n"
            f"```python\n{code_template}\n```\n\n"
            "### Answer: (use the provided format with backticks)"
        )
    ]


DATASET_SPECS = (  # 全部支持的数据集注册表；顺序即 --list-supported 的输出顺序
    DatasetSpec("gsm8k", "openai/gsm8k", "main", "test", format_gsm8k),  # 小学数学应用题
    DatasetSpec("math500", "HuggingFaceH4/MATH-500", None, "test", format_math_problem),  # MATH 竞赛数学 500 题子集
    DatasetSpec("aime24", "HuggingFaceH4/aime_2024", None, "train", format_math_problem),  # 2024 AIME 竞赛题
    DatasetSpec("aime25", "MathArena/aime_2025", None, "train", format_math_problem),  # 2025 AIME 竞赛题（30 题）
    DatasetSpec("alpaca", "tatsu-lab/alpaca", None, "train", format_alpaca),  # 指令跟随问答
    DatasetSpec("mt-bench", "HuggingFaceH4/mt_bench_prompts", None, "train", format_mt_bench),  # 多轮对话提问
    DatasetSpec("humaneval", "openai/openai_humaneval", None, "test", format_humaneval),  # Python 代码生成
    DatasetSpec("mbpp", "google-research-datasets/mbpp", "sanitized", "test", format_mbpp),  # 基础 Python 编程（sanitized 子集）
    DatasetSpec(  # lbpp：数据集以 parquet 原始文件分发，须手工下载解析
        "lbpp",
        "CohereLabs/lbpp",
        None,
        "test",
        format_lbpp,
        parquet_files=("python/test.parquet",),
    ),
    DatasetSpec("swe-bench", "princeton-nlp/SWE-bench_Lite", None, "test", format_swe_bench),  # GitHub issue 修复
    DatasetSpec(  # livecodebench：数据集以多个原始 JSONL 分发，须手工下载解析
        "livecodebench",
        "livecodebench/code_generation_lite",
        None,
        "test",
        format_livecodebench,
        jsonl_files=(  # 各版本发布期的题目文件，全部合并转换
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ),
    ),
)
DATASET_SPECS_BY_HF_NAME = {spec.hf_name: spec for spec in DATASET_SPECS}  # HF 名 → spec 的查找表


def parse_args() -> argparse.Namespace:
    # 解析转换命令行参数：数据集名（位置参数）+ 各类覆盖/控制选项。
    parser = argparse.ArgumentParser(
        description="Convert a supported Hugging Face eval dataset to turns JSONL."
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",  # 可选位置参数：--list-supported 时不传
        help="Hugging Face dataset name, for example `openai/gsm8k`.",
    )
    parser.add_argument(
        "--config-name",  # 覆盖注册表中的 HF config/子集名
        help="Override the Hugging Face dataset config/subset name.",
    )
    parser.add_argument(
        "--split",  # 覆盖注册表中的 split
        help="Override the Hugging Face split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,  # 默认写到 eval_datasets/
        help="Directory for the default output file.",
    )
    parser.add_argument(
        "--output-path",  # 显式输出路径，优先于 --output-root
        type=Path,
        help="Exact output JSONL path. Overrides --output-root.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Convert at most this many rows; useful for smoke tests.",  # 限制行数，用于快速冒烟
    )
    parser.add_argument(
        "--list-supported",
        action="store_true",
        help="Print supported Hugging Face dataset names and exit.",  # 只打印支持列表即退出
    )
    return parser.parse_args()


def print_supported_datasets() -> None:
    # 以制表符分隔打印全部支持数据集：HF 名、输出文件名、config、split。
    for spec in DATASET_SPECS:
        config = spec.config_name if spec.config_name is not None else "-"  # 无 config 显示占位符
        print(
            f"{spec.hf_name}\toutput={spec.output_name}.jsonl\t"
            f"config={config}\tsplit={spec.split}"
        )


def require_dataset_spec(dataset_name: str) -> DatasetSpec:
    # 按 HF 名查注册表；未登记的数据集直接报错并列出全部支持名。
    if dataset_name in DATASET_SPECS_BY_HF_NAME:
        return DATASET_SPECS_BY_HF_NAME[dataset_name]

    supported = ", ".join(spec.hf_name for spec in DATASET_SPECS)
    raise ValueError(
        f"Unsupported Hugging Face dataset name: {dataset_name}. "
        f"Supported names: {supported}"
    )


def load_hf_dataset(spec: DatasetSpec, args: argparse.Namespace):
    # 常规路径：用 datasets.load_dataset 拉取并返回 Iterable/普通 Dataset。
    from datasets import load_dataset  # 延迟导入：仅实际转换时才需要装 datasets

    config_name = args.config_name if args.config_name is not None else spec.config_name  # CLI 覆盖优先
    split = args.split if args.split is not None else spec.split  # CLI 覆盖优先

    load_kwargs = {"split": split}  # 只加载指定 split，避免整仓下载

    if config_name is None:
        return load_dataset(spec.hf_name, **load_kwargs)  # 无 config 的数据集
    return load_dataset(spec.hf_name, config_name, **load_kwargs)  # 带 config/子集的数据集


def iter_hf_jsonl_rows(spec: DatasetSpec, args: argparse.Namespace):
    # 原始 JSONL 路径：绕过 load_dataset，逐个从 HF 仓库下载文件并按行 yield。
    from huggingface_hub import hf_hub_download  # 延迟导入：仅实际转换时才需要

    if args.config_name is not None:
        raise ValueError(f"{spec.hf_name} does not use a Hugging Face config name.")
    split = args.split if args.split is not None else spec.split
    if split != spec.split:
        raise ValueError(f"{spec.hf_name} only supports split={spec.split!r}.")  # 原始文件路径固定对应某个 split，不允许换

    for filename in spec.jsonl_files:
        path = Path(  # hf_hub_download 落盘到本地缓存并返回文件路径
            hf_hub_download(spec.hf_name, filename=filename, repo_type="dataset")
        )
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()  # 去掉行尾换行与空白
                if line:
                    yield json.loads(line)  # 跳过空行后逐行解析


def iter_hf_parquet_rows(spec: DatasetSpec, args: argparse.Namespace):
    # 原始 parquet 路径：下载 parquet 文件，用 pyarrow 读表并逐行 yield dict。
    import pyarrow.parquet as pq  # 延迟导入：parquet 解析依赖
    from huggingface_hub import hf_hub_download

    if args.config_name is not None:
        raise ValueError(f"{spec.hf_name} does not use a Hugging Face config name.")
    split = args.split if args.split is not None else spec.split
    if split != spec.split:
        raise ValueError(f"{spec.hf_name} only supports split={spec.split!r}.")  # parquet 固定对应注册表里的 split

    for filename in spec.parquet_files:
        path = hf_hub_download(spec.hf_name, filename=filename, repo_type="dataset")
        yield from pq.read_table(path).to_pylist()  # 整表读入内存后逐行产出


def iter_dataset_rows(spec: DatasetSpec, args: argparse.Namespace):
    # 行迭代统一入口：按 spec 声明的来源类型（原始 JSONL / parquet / load_dataset）分发。
    if spec.jsonl_files:
        yield from iter_hf_jsonl_rows(spec, args)
        return
    if spec.parquet_files:
        yield from iter_hf_parquet_rows(spec, args)
        return

    yield from load_hf_dataset(spec, args)  # 默认走 load_dataset


def resolve_output_path(spec: DatasetSpec, args: argparse.Namespace) -> Path:
    # 输出路径解析：显式 --output-path 优先，否则 <output-root>/<output_name>.jsonl。
    if args.output_path is not None:
        return args.output_path
    return args.output_root / f"{spec.output_name}.jsonl"


def validate_turns(turns: list[str], row_number: int) -> None:
    # 写盘前校验：turns 必须非空且每轮都是非空字符串，否则报错并给出行号。
    if not turns:
        raise ValueError(f"row {row_number} produced no turns.")
    for turn in turns:
        if not isinstance(turn, str) or not turn:
            raise ValueError(f"row {row_number} produced an invalid turn: {turn!r}")


def convert_dataset(spec: DatasetSpec, args: argparse.Namespace) -> tuple[Path, int]:
    # 核心转换：逐行取数 → 格式化 → 校验 → 写 {"turns": [...]} JSONL；返回输出路径与行数。
    output_path = resolve_output_path(spec, args)
    if output_path.exists():
        raise FileExistsError(f"Output JSONL already exists: {output_path}")  # 默认拒绝覆盖，防误毁已有评测集

    output_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录不存在则创建

    count = 0  # 已写出行数
    with output_path.open("w", encoding="utf-8") as handle:
        for row_number, row in enumerate(iter_dataset_rows(spec, args), start=1):  # 行号从 1 起，便于报错定位
            if args.max_rows is not None and count >= args.max_rows:
                break  # 达到 --max-rows 上限即停

            turns = spec.format_turns(row)  # 按数据集各自的回调格式化
            validate_turns(turns, row_number)  # 校验通过才落盘
            handle.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")  # 统一写 {"turns": [...]} 行格式
            count += 1

    return output_path, count


def main() -> None:
    # CLI 入口：--list-supported 只打印支持列表；否则查 spec 并执行转换。
    args = parse_args()
    if args.list_supported:
        print_supported_datasets()
        return
    if args.dataset_name is None:
        raise ValueError("dataset_name is required unless --list-supported is used.")

    spec = require_dataset_spec(args.dataset_name)
    try:
        output_path, count = convert_dataset(spec, args)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None  # 目标文件已存在：仅打印消息并以非零码退出，不打印堆栈
    print(f"wrote {spec.hf_name}: {count} rows -> {output_path}")


if __name__ == "__main__":
    main()
