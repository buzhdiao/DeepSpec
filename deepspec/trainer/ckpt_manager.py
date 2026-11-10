import os  # 路径拼接、symlink 检测、realpath
import random  # Python 内置 RNG 状态的保存/恢复（四路 RNG 之一）
import shutil  # 复制训练配置文件进 checkpoint
from dataclasses import dataclass  # 声明式数据类

import numpy as np  # NumPy RNG 状态保存/恢复
import torch  # torch.load/save：训练状态（优化器+RNG）的序列化
import torch.distributed as dist  # barrier：各 rank 保存节奏同步
import yaml  # 解析命令行覆盖项的原始值（safe_load 还原类型）
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 判断/操作 FSDP 封装模型
from torch.distributed.fsdp import FullStateDictConfig, StateDictType  # 聚合完整 state_dict 的配置

from deepspec.utils import (  # 目录、symlink、分布式日志工具
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
    safe_symlink,  # 原子符号链接更新（os.replace 中转）
)


TRAIN_CONFIG_FILE_NAME = "train_config.py"  # 每个 checkpoint 内固定的配置文件名


def discover_latest_checkpoint(checkpoint_dir):
    # 定位最近一次 checkpoint：checkpoint 根目录下的 step_latest 符号链接
    # 指向 step_N 子目录；存在（链接或目录均可）则返回其真实路径，否则返回 None（从零训练）。
    latest_link = os.path.join(checkpoint_dir, "step_latest")
    if not (os.path.islink(latest_link) or os.path.isdir(latest_link)):
        return None
    return os.path.realpath(latest_link)  # 解析链接的真实目标（跨改动稳定）


def save_train_config(*, train_config, checkpoint_dir: str) -> str:
    # 把训练配置快照进 checkpoint：先复制配置原始文件，再追加命令行覆盖项，
    # 使任何一次实验都能从 checkpoint 本身完整还原出当时的配置。
    # 仅全局 rank0 执行；其他 rank 直接返回目标路径（文件已由 rank0 落盘）。
    dest_path = os.path.join(checkpoint_dir, TRAIN_CONFIG_FILE_NAME)
    if not is_global_main_process():
        return dest_path

    ensure_dir(checkpoint_dir)
    shutil.copy(train_config._origin_config_path, dest_path)  # 复制解析前的原始配置文件
    opts = train_config._origin_opts  # 命令行 --opts 覆盖项列表（"key.path=value" 形式）
    if opts:
        with open(dest_path, "a", encoding="utf-8") as handle:
            handle.write("\n\n# --opts overrides applied at save time\n")
            for opt in opts:
                handle.write(_render_opt_assignment(opt) + "\n")  # 追加为可执行赋值语句
    return dest_path


def _render_opt_assignment(opt: str) -> str:
    # 把覆盖项字符串 "train.lr=1e-4" 还原成配置文件语句 "train['lr'] = 0.0001"。
    # key 按点号切分：首段保留为对象名，后续段渲染成 ['attr'] 下标链。
    # 值经 yaml.safe_load 还原类型（数字/布尔/字符串），再 repr 输出。
    key, raw_value = opt.split("=", 1)
    head, *rest = key.split(".")
    accessors = "".join(f"[{part!r}]" for part in rest)
    value = yaml.safe_load(raw_value)
    return f"{head}{accessors} = {value!r}"


@dataclass(frozen=True)
class TrainingResumeState:
    # next_micro_step is the single source of truth for training progress;
    # global_step and current_epoch are derived from it together with
    # gradient_accumulation_steps / micro_batches_per_epoch.
    # 续训状态载体：micro 步计数是唯一进度真值，全局步/epoch 均可由它派生，
    # 避免多个计数器间出现不一致。frozen=True：加载后不可再改。
    next_micro_step: int


def load_resume_draft_model(
    *,
    resume_checkpoint_dir: str,  # step_latest 指向的 checkpoint 目录（内含 HF 格式权重）
    draft_model,  # 已构建（嵌入已继承）的 draft 模型：仅用于取类型与配置
    device,  # 本 rank 的 CUDA 设备
    precision_dtype,  # 加载精度（与训练一致）
    global_rank: int,  # 校验该 rank 的训练状态文件存在
):
    # 从 checkpoint 恢复 draft 模型权重：以 HF from_pretrained 方式加载 step_N 目录。
    # attn_implementation 保持与原模型一致，避免恢复后注意力实现悄悄变化。
    # 注意：该方法只恢复权重；优化器/进度/RNG 由 load_training_state 负责。
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)
    resumed_model = type(draft_model).from_pretrained(  # 同类型模型从 checkpoint 目录加载
        resume_checkpoint_dir,
        dtype=precision_dtype,
        attn_implementation=str(draft_model.config._attn_implementation),  # 锁定注意力实现
    )
    resumed_model = resumed_model.to(device=device, dtype=precision_dtype)
    resumed_model.set_embedding_head_trainable(False)  # 嵌入/lm_head 重新冻结（与全新构建一致）
    return resumed_model


def load_training_state(
    *,
    resume_checkpoint_dir: str,
    optimizer,  # 已构建的 BF16Optimizer：状态字典就地恢复
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
    micro_batches_per_epoch: int,
) -> TrainingResumeState:
    # 恢复训练状态：优化器参数、micro 步进度与四路 RNG（torch/cuda/numpy/python）。
    # RNG 状态逐 rank 保存与恢复：续训后每个 rank 的数据增强/丢弃等随机行为与暂停前完全一致。
    # 强校验拓扑参数（rank/world_size/local_batch_size）与进度对齐——
    # 改并发规模或批大小续训会破坏数据切分与调度器状态，必须显式禁止。
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)

    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)  # 整包载入 CPU（weights_only=False：含非张量状态）
    optimizer.load_state_dict(checkpoint["optimizer"])  # 恢复 AdamW 状态与调度器进度

    next_micro_step = int(checkpoint["next_micro_step"])
    assert next_micro_step % gradient_accumulation_steps == 0, (  # 进度必须落在累积窗口边界（checkpoint 只在窗口末保存）
        "next_micro_step must be aligned with gradient_accumulation_steps."
    )

    saved_rank = int(checkpoint["global_rank"])
    assert saved_rank == int(global_rank)
    
    saved_world_size = int(checkpoint["world_size"])
    assert saved_world_size == int(world_size)
    
    saved_local_batch_size = int(checkpoint["local_batch_size"])
    assert saved_local_batch_size == int(local_batch_size)

    torch.set_rng_state(checkpoint["torch_rng"])  # CPU RNG 状态
    torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"])  # 当前 GPU RNG 状态
    np.random.set_state(checkpoint["numpy_rng"])  # NumPy RNG 状态
    random.setstate(checkpoint["python_rng"])  # Python 内置 RNG 状态

    global_step = next_micro_step // gradient_accumulation_steps  # 派生全局步（仅日志用）
    current_epoch = next_micro_step // micro_batches_per_epoch + 1  # 派生当前 epoch（1 起，仅日志用）
    print_on_global_main(
        (
            "AUTO-RESUME from "
            f"{resume_checkpoint_dir}, next_micro_step={next_micro_step}, "
            "to force fresh run change exp_name or remove step_latest"  # 想重训需换目录或删链接
        )
    )
    print_on_local_main(
        f"Resumed from {resume_checkpoint_dir}: "
        f"next_micro_step={next_micro_step}, global_step={global_step}, "
        f"epoch={current_epoch}"
    )
    return TrainingResumeState(next_micro_step=next_micro_step)


def save_checkpoint(
    *,
    model,  # FSDP（可能含 compile 前缀）封装的训练模型
    draft_model,  # 未封装 draft 模型：提供 save_pretrained 所需的模型元信息
    optimizer,
    checkpoint_dir_root: str,  # checkpoint 根目录
    train_config,  # 训练配置对象（含 _origin_config_path/_origin_opts）
    next_micro_step: int,  # 保存时刻的 micro 步进度
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
) -> str:
    # 保存一次完整 checkpoint（各 rank 协同）：
    # 1) rank0 建 step_N 目录并快照配置 → barrier；
    # 2) 全员聚合 FSDP 完整 state_dict，rank0 以 HF 格式落盘；
    # 3) 各 rank 各自保存 training_state.rankN.pt（优化器+进度+RNG）→ barrier；
    # 4) rank0 原子更新 step_latest 符号链接指向本目录 → barrier。
    # barrier 划分阶段：保证 rank0 读 state_dict / 更新链接时其他 rank 已就绪。
    assert next_micro_step % gradient_accumulation_steps == 0, (  # 只允许在累积窗口边界保存
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    global_step = next_micro_step // gradient_accumulation_steps
    checkpoint_dir = os.path.join(checkpoint_dir_root, f"step_{global_step}")  # 目录名即步数
    if is_global_main_process():
        ensure_dir(checkpoint_dir)
        save_train_config(train_config=train_config, checkpoint_dir=checkpoint_dir)  # 配置快照
    dist.barrier()
    _save_model_checkpoint(
        model=model,
        draft_model=draft_model,
        checkpoint_dir=checkpoint_dir,
    )
    training_state = _serialize_training_state(
        optimizer=optimizer,
        next_micro_step=next_micro_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_rank=global_rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
    )
    torch.save(  # 每 rank 独立训练状态：RNG/优化器状态本就按 rank 不同
        training_state,
        _rank_training_state_path(checkpoint_dir, global_rank),
    )
    dist.barrier()  # 全部权重+状态落盘完成
    if is_global_main_process():
        safe_symlink(  # 原子更新 step_latest：恢复端永远看到完整一致的新 checkpoint
            checkpoint_dir,
            os.path.join(checkpoint_dir_root, "step_latest"),
        )
        print_on_global_main(f"Saved checkpoint to {checkpoint_dir}")
    dist.barrier()  # 链接更新完成后才放行（返回调用方提交评估）
    return checkpoint_dir


def _rank_training_state_path(checkpoint_dir: str, global_rank: int) -> str:
    # 某 rank 训练状态文件路径：training_state.rankN.pt（每 rank 一份，互不覆盖）。
    return os.path.join(
        checkpoint_dir,
        f"training_state.rank{int(global_rank)}.pt",
    )


def _serialize_training_state(
    *,
    optimizer,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
):
    # 打包训练状态：进度真值 + 优化器状态 + 拓扑参数（续训校验用）+ 四路 RNG 状态。
    # 保存 RNG 是无偏差续训的关键：数据顺序/丢弃等随机源在恢复后逐位一致。
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    return {
        "next_micro_step": int(next_micro_step),
        "optimizer": optimizer.state_dict(),
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "torch_rng": torch.get_rng_state(),  # CPU RNG
        "torch_cuda_rng": torch.cuda.get_rng_state(),  # 当前 GPU RNG
        "numpy_rng": np.random.get_state(),  # NumPy RNG
        "python_rng": random.getstate(),  # Python 内置 RNG
    }


def _full_model_state_dict(model):
    # 从 FSDP 封装模型聚合"完整 state_dict"：各 rank 的分片参数 all-gather 成
    # 全量参数并 offload 到 CPU；rank0_only=True 时仅 rank0 拿到完整字典，
    # 其他 rank 得到空 dict——省去非 0 rank 的落盘与内存峰值。
    assert isinstance(model, FSDP), "training model must be wrapped in FSDP"
    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(  # 上下文切换 state_dict 导出形态
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config,
    ):
        return model.state_dict()


def _save_model_checkpoint(*, model, draft_model, checkpoint_dir: str):
    # 落盘模型权重：聚合完整 state_dict，剥离 torch.compile 注入的 "_orig_mod." 前缀
    # 使键名与原始 draft 模型对齐，再走 HF save_pretrained（safetensors+config）。
    # 从 HF checkpoint 直接 from_pretrained 即可无差别恢复。
    state_dict = _full_model_state_dict(model)
    if is_global_main_process():
        draft_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key
            if normalized_key.startswith("_orig_mod."):
                normalized_key = normalized_key[len("_orig_mod.") :]  # 剥离 compile 包装前缀
            draft_state_dict[normalized_key] = value
        assert draft_state_dict, "Failed to extract draft model state_dict from checkpoint."
        draft_model.save_pretrained(checkpoint_dir, state_dict=draft_state_dict)  # HF 格式落盘
