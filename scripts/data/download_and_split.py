from __future__ import annotations  # 延迟注解求值：允许注解使用 X | None 等新语法

import argparse  # 命令行参数解析
import json  # 写训练/评测 JSONL
from pathlib import Path  # 路径对象：输出目录/文件处理


DEFAULT_DATASET_NAME = "mlabonne/open-perfectblend"  # 默认源数据集：PerfectBlend 大规模对话混合集
DEFAULT_TRAIN_OUTPUT_PATH = Path("cache/dataset/perfectblend_train.jsonl")  # 训练对话样本默认输出路径
DEFAULT_TEST_OUTPUT_DIR = Path("eval_datasets")  # 评测 turns 文件默认输出目录
DEFAULT_TEST_OUTPUT_NAME = "perfectblend.jsonl"  # 评测 turns 默认文件名（任务名 perfectblend）
ROLE_MAPPING = {  # PerfectBlend 的 from 角色 → 统一 role 名的映射
    "human": "user",  # 人类发言
    "gpt": "assistant",  # GPT 回复
    "chatgpt": "assistant",  # ChatGPT 回复
    "bing": "assistant",  # Bing 聊天回复
    "bard": "assistant",  # Bard 回复
}


def parse_args() -> argparse.Namespace:
    # 解析下载与切分参数：源数据集、切分比例与两路输出路径。
    parser = argparse.ArgumentParser(
        description="Download PerfectBlend and split it into train and eval JSONL files."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,  # 可换源数据集，默认 PerfectBlend
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",  # HF 上 PerfectBlend 只有 train split
        help="Hugging Face split to load.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Process only the first N source rows before splitting.",  # 只取前 N 行，快速跑通流程
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.05,
        help="Fraction held out for eval, matching SpecForge's default.",  # 5% 切给评测，对齐 SpecForge 约定
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by train_test_split.",  # 切分随机种子，保证两次运行切分一致
    )
    parser.add_argument(
        "--train-output-path",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT_PATH,
        help="Output path for training conversations JSONL.",
    )
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        default=DEFAULT_TEST_OUTPUT_DIR,
        help="Directory for the eval turns JSONL file.",
    )
    parser.add_argument(
        "--test-output-name",
        default=DEFAULT_TEST_OUTPUT_NAME,
        help="Eval turns JSONL filename under --test-output-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip only when both expected output files already exist.",  # 两路输出都已存在时整体跳过（幂等重跑）
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    # 参数合法性校验：抽样数须为正，切分比例须在开区间 (0,1)。
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be in the range (0, 1).")


def test_output_path(args: argparse.Namespace) -> Path:
    # 拼出评测 turns 文件的完整路径。
    return args.test_output_dir / args.test_output_name


def check_output_paths(args: argparse.Namespace) -> bool:
    # 输出冲突检查：任何输出已存在即报错；--skip-existing 且全部存在则整体跳过。返回 True 表示可跳过。
    output_paths = [args.train_output_path, test_output_path(args)]  # 两个预期输出
    existing_paths = [path for path in output_paths if path.exists()]  # 其中已存在的
    if not existing_paths:
        return False
    if args.skip_existing and len(existing_paths) == len(output_paths):  # 两个输出齐全才允许整体跳过
        for path in output_paths:
            print(f"skip existing output: {path}")
        return True

    joined_paths = ", ".join(str(path) for path in existing_paths)
    raise FileExistsError(f"Output JSONL already exists: {joined_paths}")  # 部分存在：拒绝，防覆盖半成品


def add_index(row: dict, idx: int) -> dict:
    # map 回调：给每行写入全局序号 id，保证切分后两个文件可对齐溯源。
    row["id"] = idx
    return row


def load_source_dataset(args: argparse.Namespace):
    # 拉取源数据集：可选截断前 N 行，再给每行编号。
    from datasets import load_dataset  # 延迟导入：仅实际运行时才需要 datasets

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.sample_size is not None and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))  # 截取前 N 行（--sample-size）
    dataset = dataset.map(add_index, with_indices=True)  # 逐行写入行号 id
    return dataset


def normalize_conversations(row: dict) -> dict:
    # 把 PerfectBlend 的 conversations 规范化：from/value → role/content，丢弃映射表之外的异常角色。
    conversations = []
    for message in row["conversations"]:
        if message["from"] not in ROLE_MAPPING:
            continue  # 未登记的角色（如 system/tool）直接丢弃
        conversations.append(
            {
                "role": ROLE_MAPPING[message["from"]],  # 统一为 user/assistant
                "content": message["value"],  # 消息正文
            }
        )
    return {"id": row["id"], "conversations": conversations}  # 保留 id + 规范化对话


def validate_conversations(row: dict, row_number: int) -> None:
    # 写盘前校验：对话非空、以 user 开头、角色合法、每条内容为非空字符串。
    conversations = row["conversations"]
    if not conversations:
        raise ValueError(f"row {row_number} produced no conversations.")
    if conversations[0]["role"] != "user":
        raise ValueError(f"row {row_number} does not start with a user message.")  # 训练/生成流程都假定首条为 user
    for message in conversations:
        if message["role"] not in {"user", "assistant"}:
            raise ValueError(f"row {row_number} has invalid role: {message['role']}")
        if not isinstance(message["content"], str) or not message["content"]:
            raise ValueError(f"row {row_number} has invalid message content.")


def user_turns(row: dict) -> list[str]:
    # 抽取全部 user 消息内容作为 turns（评测只看 user 侧提问）。
    return [
        message["content"]
        for message in row["conversations"]
        if message["role"] == "user"
    ]


def write_train_jsonl(dataset, output_path: Path) -> int:
    # 写训练文件：每行 {"id", "conversations"}（规范化后的完整多轮对话）。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录不存在则创建
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row_number, row in enumerate(dataset, start=1):  # 行号从 1 起，便于报错定位
            converted = normalize_conversations(row)  # 规范化角色字段
            validate_conversations(converted, row_number)  # 校验通过才落盘
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_eval_jsonl(dataset, output_path: Path) -> int:
    # 写评测文件：每行 {"turns": [...]}，只保留 user 轮，格式对齐 eval_datasets 其余任务。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row_number, row in enumerate(dataset, start=1):
            converted = normalize_conversations(row)
            validate_conversations(converted, row_number)
            turns = user_turns(converted)  # 只留 user 消息
            handle.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    # 主流程：参数校验 → 输出冲突检查 → 下载/编号 → 按比例切分 → 分别写训练与评测文件。
    args = parse_args()
    validate_args(args)
    try:
        if check_output_paths(args):  # 输出已存在且允许跳过时直接返回
            return
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None  # 输出冲突：仅打印消息并以非零码退出，不打印堆栈

    dataset = load_source_dataset(args)
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)  # 打乱并切出训练/评测两份

    train_count = write_train_jsonl(split_dataset["train"], args.train_output_path)
    eval_output_path = test_output_path(args)
    test_count = write_eval_jsonl(split_dataset["test"], eval_output_path)

    print(f"wrote train split: {train_count} rows -> {args.train_output_path}")
    print(f"wrote eval split: {test_count} rows -> {eval_output_path}")


if __name__ == "__main__":
    main()
