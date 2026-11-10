import argparse  # 命令行参数解析
from dataclasses import dataclass  # 声明前向结果结构体
import json  # 序列化 config/参数/summary
import os  # 环境变量、路径拼接与目录创建

import torch  # 前向推理、张量拼接与设备管理
import torch.distributed as dist  # 多卡同步：barrier、广播对象、销毁进程组
from torch.utils.data import DataLoader, Subset  # 数据加载器与本地样本区间切片
from transformers import AutoModel, AutoTokenizer  # 加载目标模型与 tokenizer

from deepspec.data import ConversationCollator  # 对话批处理：chat template 编码 + loss mask 构造
from deepspec.data.target_cache_dataset import (  # 目标缓存导出所需的全套工具
    AsyncTargetCacheWriter,  # 后台线程异步写 shard，隐藏磁盘 IO 延迟
    LocalCacheWriteSummary,  # 单 rank 导出产物的 summary 结构体
    atomic_json_dump,  # 原子写 JSON（先写临时文件再 rename，防半写文件）
    build_global_target_cache_shard_map,  # 汇总各 rank summary → 全局 shard 编号映射
    build_target_cache_manifest,  # 组装缓存 manifest（元数据 + shard 清单）
    cleanup_target_cache_tmp_dir,  # 删除 _tmp 中间产物目录
    compute_local_sample_range,  # 计算本 rank 负责的全局样本区间（均匀切分）
    finalize_target_cache_index,  # 拼接各 rank 本地索引为全局 samples.idx
    load_local_cache_write_summary,  # 读取某 rank 目录下的 summary.json
    prepare_target_cache_output_dir,  # 创建输出目录骨架（含 _tmp）
    rename_local_target_cache_shards,  # 按 shard_map 把本地 shard 改名移动到输出根目录
    write_target_cache_manifest,  # 把 manifest 落盘
)
from deepspec.data.jsonl_dataset import JsonLineDataset  # mmap 式 JSONL 数据集（O(1) 随机访问）
from deepspec.utils import (  # 通用工具集
    CustomJSONEncoder,  # 可序列化函数/类/Path/ConfigNode 的 JSON 编码器
    get_git_diff,  # 读取工作区 git diff（记录实验代码状态）
    get_git_sha,  # 读取当前 commit SHA
    init_dist,  # 初始化 NCCL 进程组，返回 device/global_rank/world_size
    is_global_main_process,  # 是否全局 rank 0（跨节点唯一）
    load_config,  # 以模块方式加载 config .py 为 ConfigNode
    main_process_first,  # 让全局 rank 0 先执行的上下文管理器（避免并发写缓存）
    parse_opts_to_config,  # 把 --opts "key.path=value" 覆盖进 config
    print_on_global_main,  # 仅全局 rank 0 打印
    print_on_local_main,  # 仅本节点 0 号卡打印
    seed_all,  # 统一设置 python/numpy/torch 随机种子
)

os.environ["USE_TORCH"] = "true"  # 环境开关：标记 torch 后端运行环境（沿用上游训练框架约定）
os.environ["WANDB_DISABLED"] = "true"  # 禁用 wandb 在线上报
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 HF tokenizer 多线程，避免 spawn 多进程下的 fork 告警/死锁

# PyTorch 2.10 Inductor still reads the legacy allow_tf32 flag while compiling.
torch.set_float32_matmul_precision("high")  # fp32 matmul 允许 TF32，降低编译路径开销


@dataclass(frozen=True)
class TargetForwardResult:
    # 一次目标模型前向的捕获结果：目标层拼接隐状态 + 最后一层隐状态。
    # frozen=True：结果在 hook 移除后即不可变，可安全向下传递。
    target_hidden_states: torch.Tensor  # 各目标层隐状态沿特征维拼接，形状 [B, L, 层数*H]
    target_last_hidden_states: torch.Tensor  # 最后一层隐状态，形状 [B, L, H]


def _get_target_backbone(target_model):
    # 取目标模型的纯文本 backbone（含 layers/embed_tokens 的模块）：兼容 Gemma4 多模态包装结构。
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):  # Gemma4 多模态模型需下钻到语言子模块
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(target_model.model, "language_model"):
            return target_model.model.language_model
        assert False, "Gemma4 target model must expose a text language_model."
    return getattr(target_model, "model", target_model)  # 常规 CausalLM：body 在 .model 下；裸 backbone 则原样返回


def _get_target_hidden_size(target_model) -> int:
    # 读目标模型单层隐状态宽度 H：Gemma4 的文本宽度藏在 text_config 里，其余直接读 hidden_size。
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        return int(target_model.config.text_config.hidden_size)
    return int(target_model.config.hidden_size)


def _get_hook_tensor(output):
    # 从 forward hook 的输出中取隐状态张量：直接是 Tensor 则原样；tuple/list 则取第一个元素。
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]  # HF 层输出多为 (hidden_states, ...) 结构，首项即隐状态
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def run_target_forward_with_hooks(
    *,
    target_model,  # 目标模型（eval 模式，权重不更新）
    input_ids: torch.Tensor,  # token id 批，形状 [B, L]
    attention_mask: torch.Tensor,  # 注意力掩码，形状 [B, L]（1=有效，0=padding）
    target_layer_ids,  # 要捕获的层号列表；-1 表示 embed_tokens 层
):
    # 用 forward hook 捕获目标层隐状态：注册 hook → no_grad 前向 → 拼接 → finally 必移除 hook。
    backbone = _get_target_backbone(target_model)  # 下钻到文本 backbone
    layer_modules = backbone.layers  # decoder 层模块列表
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured_hidden_states = {}  # 层号 → 该层隐状态（detach 后脱离计算图）
    handles = []  # hook 句柄，finally 中统一移除

    def capture_layer(layer_id: int):
        # 生成层号绑定的 hook：把该层输出 detach 后按层号存入字典。
        def hook(_module, _inputs, output):
            captured_hidden_states[layer_id] = _get_hook_tensor(output).detach()

        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(
                backbone.embed_tokens.register_forward_hook(capture_layer(-1))  # -1 约定：捕获词嵌入输出
            )
        for layer_id in target_layer_ids:
            if layer_id < 0:
                continue  # 负层号只有 -1（embed），其余跳过
            handles.append(
                layer_modules[layer_id].register_forward_hook(capture_layer(layer_id))  # 按层号注册 forward hook
            )

        with torch.no_grad():  # 只做推理导出，不建计算图不占显存梯度
            target_output = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,  # 不走 HF 内建全层收集，只取 hook 捕获的目标层，省显存
                use_cache=False,  # 无需 KV cache
            )
            target_last_hidden_states = target_output.last_hidden_state.detach()  # 最后一层隐状态，[B, L, H]
            target_hidden_states = torch.cat(
                [captured_hidden_states[layer_id] for layer_id in target_layer_ids],
                dim=-1,  # 沿特征维拼接目标层：[B, L, 层数*H]，层顺序与 target_layer_ids 一致
            )
    finally:
        for handle in handles:
            handle.remove()  # 无论成败都移除 hook，避免污染后续前向
        captured_hidden_states.clear()

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
    )


def parse_args():
    # 解析 CLI 参数并加载 config：config 定模型/层数/分词，CLI 定数据与导出行为。
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)  # 训练 config .py 路径（取模型/数据字段）
    parser.add_argument("--opts", action="append", default=[])  # 可重复的 config 字段覆盖项
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",  # 可重复：多份训练 JSONL 合并导出
    )
    parser.add_argument("--output-dir", required=True)  # 缓存输出根目录
    parser.add_argument("--min-loss-tokens", type=int, default=14)  # 样本最少 loss token 数（过滤过短对话）
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)  # 单 shard 文件上限 64GiB，超过自动切分
    parser.add_argument("--local-batch-size", type=int, default=32)  # 单卡前向批大小
    parser.add_argument("--num-workers", type=int, default=4)  # DataLoader worker 数
    cli_args = parser.parse_args()
    config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))  # 合并 CLI 覆盖并执行 finalize_cfg 钩子
    return cli_args, config


def _write_manifest(
    *,
    output_dir: str,  # 缓存输出根目录
    config,  # 训练 config（提取模型名/chat template 等元数据）
    train_data_paths,  # 源 JSONL 路径列表（写入 manifest 供溯源）
    target_layer_ids,  # 捕获层号列表（训练加载缓存时校验层配置一致）
    hidden_size: int,  # 单层隐状态宽度
    min_loss_tokens: int,  # 最少 loss token 数
    shards,  # 全局 shard 清单（shard_id + 文件名）
):
    # 汇总各 rank 样本数并写 manifest：训练侧据此校验缓存与当前 config 是否匹配。
    num_samples = sum(  # 全局有效样本总数 = 各 rank 本地样本数之和
        int(
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")  # 逐 rank 读 summary.json
            )["num_local_samples"]
        )
        for rank in range(dist.get_world_size())
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={  # 附加溯源/校验元数据
            "target_model_name_or_path": str(config.model.target_model_name_or_path),  # 目标模型名：换模型必须重建缓存
            "source_jsonl_paths": [str(path) for path in train_data_paths],  # 源数据路径
            "chat_template": str(config.data.chat_template),  # chat template 名
            "max_length": int(config.data.max_length),  # 编码最大长度
            "min_loss_tokens": int(min_loss_tokens),
            "project_name": (  # 项目名（可空）
                str(config.get("project_name"))
                if config.get("project_name") is not None
                else None
            ),
            "exp_name": (  # 实验名（可空）
                str(config.get("exp_name"))
                if config.get("exp_name") is not None
                else None
            ),
            "git_sha": str(get_git_sha()),  # 生成缓存时的代码版本
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)


def _print_prepare_progress(*, global_rank: int, processed_samples: int, total_samples: int):
    # 打印单 rank 导出进度（每 ≥100 条或收尾时调用一次）。
    print(
        f"[prepare rank {global_rank}] {processed_samples}/{total_samples} samples",
        flush=True,
    )


def main(local_rank: int):
    # 单卡 worker 主流程：分区读样本 → 目标模型前向捕获隐状态 → 异步写本地 shard → 全局合并索引/manifest。
    cli_args, config = parse_args()
    train_data_paths = list(cli_args.train_data_path)  # 多份训练 JSONL 路径
    target_layer_ids = [int(layer_id) for layer_id in config.model.target_layer_ids]  # 目标层号（如 [1, 9, 17, 25, 33]）
    min_loss_tokens = int(cli_args.min_loss_tokens)
    seed_all(int(config.seed))
    device, global_rank, world_size = init_dist(local_rank)  # 初始化 NCCL 进程组（无 torchrun 环境变量时按单机本地组）
    output_dir = os.path.abspath(cli_args.output_dir)
    print_on_local_main(json.dumps(config, indent=4, cls=CustomJSONEncoder), flush=True)  # 本节点 0 号卡打印最终 config
    print_on_local_main(
        json.dumps(
            {
                "train_data_path": train_data_paths,
                "output_dir": output_dir,
                "target_layer_ids": target_layer_ids,
                "min_loss_tokens": min_loss_tokens,
                "max_shard_bytes": int(cli_args.max_shard_bytes),
                "local_batch_size": int(cli_args.local_batch_size),
                "num_workers": int(cli_args.num_workers),
            },
            indent=4,
        ),
        flush=True,
    )
    if global_rank == 0:
        prepare_target_cache_output_dir(output_dir)  # 仅全局 rank 0 创建输出目录骨架
    dist.barrier()  # 等目录就绪后其他 rank 才继续

    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")  # 本 rank 中间产物目录
    os.makedirs(rank_dir, exist_ok=True)

    with main_process_first():  # 全局 rank 0 先建行索引缓存，其余 rank 等 barrier 后直接命中缓存
        dataset = JsonLineDataset(data_paths=train_data_paths)

    local_start, local_end = compute_local_sample_range(  # 均匀切分：本 rank 负责的全局样本区间 [start, end)
        num_samples=len(dataset),
        rank=global_rank,
        world_size=world_size,
    )
    local_total_samples = local_end - local_start

    local_subset = Subset(dataset, range(local_start, local_end))  # 只取本 rank 区间，无需分布式 sampler
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path,  # 用目标模型自带的 tokenizer
    )
    target_model = AutoModel.from_pretrained(
        config.model.target_model_name_or_path,
        dtype=torch.bfloat16,  # bf16 前向：省显存且隐状态精度足够
        attn_implementation="sdpa",  # 用 PyTorch SDPA 注意力（无需 flash-attn 依赖）
    ).to(device=device).eval()  # 推理模式：关闭 dropout
    target_hidden_size = _get_target_hidden_size(target_model)  # 单层隐状态宽度 H，写进 manifest
    train_collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=config.data.chat_template,  # 与训练一致的 chat template
        max_length=config.data.max_length,  # 编码截断长度
        min_loss_tokens=min_loss_tokens,  # 过短样本在 collator 中被过滤
    )
    dataloader = DataLoader(
        local_subset,
        batch_size=int(cli_args.local_batch_size),
        collate_fn=train_collator,
        num_workers=int(cli_args.num_workers),
        pin_memory=True,  # 锁页内存加速 H2D 拷贝（配合 non_blocking）
        drop_last=False,  # 不丢尾部不满批，保证样本全量导出
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),  # 单 shard 字节上限
        max_queue_size=int(cli_args.local_batch_size) * 4,  # 写队列深度：批大小 × 4，做生产者-消费者缓冲
    )

    processed_local_samples = 0  # 已处理样本数（按批推进，不精确到单条）
    last_progress_printed = 0  # 上次打印进度时的样本数（每 ≥100 条打印一次）
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                processed_local_samples = min(
                    (batch_idx + 1) * int(cli_args.local_batch_size),
                    local_total_samples,  # 末批不满时截到实际总数
                )
                should_print_progress = (
                    processed_local_samples - last_progress_printed >= 100
                    or processed_local_samples == local_total_samples  # 收尾一批也打印
                )
                if batch is None:
                    # 整批被 collator 过滤（全部短于 min_loss_tokens）：无内容可导出，仅推进进度。
                    if should_print_progress:
                        _print_prepare_progress(
                            global_rank=global_rank,
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                batch = {
                    key: value.to(device, non_blocking=True)  # 张量异步搬上 GPU
                    for key, value in batch.items()
                }
                target_result = run_target_forward_with_hooks(
                    target_model=target_model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_layer_ids=target_layer_ids,
                )
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()  # 各样本真实长度（去掉 padding）
                for sample_idx_in_batch, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    writer.write_sample(  # 逐样本截掉 padding 后写入异步写线程
                        input_ids=batch["input_ids"][sample_idx_in_batch, :seq_len],
                        attention_mask=batch["attention_mask"][
                            sample_idx_in_batch, :seq_len
                        ],
                        loss_mask=batch["loss_mask"][sample_idx_in_batch, :seq_len],  # 仅 assistant 段为 1
                        target_hidden_states=target_result.target_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                        target_last_hidden_states=target_result.target_last_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                    )
                if should_print_progress:
                    _print_prepare_progress(
                        global_rank=global_rank,
                        processed_samples=processed_local_samples,
                        total_samples=local_total_samples,
                    )
                    last_progress_printed = processed_local_samples
    finally:
        writer.close()  # 无论成败都关写线程：flush 队列并校验无丢样本
    del target_model  # 释放显存，给后续收集阶段留空间
    torch.cuda.empty_cache()
    dataset.close()  # 关闭 mmap 与文件句柄
    summary = LocalCacheWriteSummary(
        global_rank=global_rank,
        source_sample_start=local_start,  # 本 rank 负责的全局样本区间起点
        source_sample_end=local_end,  # 区间终点（不含）
        num_local_samples=writer.num_local_samples,  # 本 rank 实际导出的样本数（可能少于区间：过滤了过短样本）
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),  # 本地 shard 文件名列表
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))  # 原子写 summary，防半写
    dist.barrier()  # 等全部 rank 的 shard/summary 落盘完成

    shard_map = None  # 全局 rank → shard 编号列表 的映射
    summaries = None  # 各 rank summary 列表（仅 rank0 加载）
    if is_global_main_process():
        summaries = [
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )
            for rank in range(world_size)
        ]
        shard_map, shards = build_global_target_cache_shard_map(summaries)  # 按源样本区间排序，给全部 shard 分配全局编号
    broadcast_payload = [shard_map]
    dist.broadcast_object_list(broadcast_payload, src=0)  # rank0 把 shard 分配结果广播给全体
    shard_map = broadcast_payload[0]
    local_summary = load_local_cache_write_summary(rank_dir)
    rename_local_target_cache_shards(  # 按 shard_map 把本地 shard 改名移动到输出根目录 shard-XXXXX.bin
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=local_summary,
        shard_map=shard_map,
    )
    dist.barrier()  # 等 shard 全部就位后 rank0 才拼全局索引

    if is_global_main_process():
        assert summaries is not None
        num_valid_samples = finalize_target_cache_index(  # 按源顺序拼接各 rank 本地索引 → 全局 samples.idx
            output_dir=output_dir,
            summaries=summaries,
            shard_map=shard_map,
        )
        _write_manifest(  # 写 manifest（样本数/层配置/模型名/元数据）
            output_dir=output_dir,
            config=config,
            train_data_paths=train_data_paths,
            target_layer_ids=target_layer_ids,
            hidden_size=target_hidden_size,
            min_loss_tokens=min_loss_tokens,
            shards=shards,
        )
        cleanup_target_cache_tmp_dir(output_dir)  # 清掉 _tmp 中间产物
        print_on_global_main(
            f"Prepared target cache at {output_dir} with "
            f"{num_valid_samples}/{len(dataset)} valid samples."  # 有效样本数可能少于总数（过短样本被过滤）
        )
    dist.barrier()  # 收尾同步：确保清理完成后再销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))  # 打印 commit SHA 及远端/分支，标记实验代码版本
        print("git diff:", get_git_diff())  # 打印未提交改动，保证缓存可复现
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())  # 每张可见 GPU 拉起一个 main 子进程（单机多卡）
