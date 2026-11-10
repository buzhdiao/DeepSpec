from contextlib import nullcontext  # 空上下文管理器：梯度累积时占位替代 no_sync
import math  # ceil：由 max_train_steps 反推 epoch 数
import os  # os.system 提交自动评估任务

import torch  # 张量与 torch.compile
import torch.distributed as dist  # NCCL 进程组、barrier 同步
from torch.distributed.device_mesh import init_device_mesh  # 混合分片时的二维设备网格
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 全分片数据并行封装
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy  # 混合精度配置与分片策略枚举
from torch.utils.data import DataLoader  # 训练数据加载
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # 加载 target 模型/词表/配置

from deepspec.data import CacheDataset, validate_train_cache  # 目标缓存数据集与其一致性校验
from deepspec.data.cuda_prefetcher import CUDAPrefetcher  # CUDA 双流预取：H2D 与计算重叠
from deepspec.utils import (  # 分布式初始化、日志、无状态可续训 sampler、BF16 优化器等工具
    BF16Optimizer,  # 封装 AdamW + warmup/两阶段调度，内部维护主权重精度策略
    StatelessResumableDistributedSampler,  # 可按样本偏移续训的分布式采样器（断点续训关键）
    ensure_dir,  # 幂等建目录
    init_dist,  # 初始化 NCCL 进程组，返回 (device, rank, world_size)
    is_global_main_process,  # 是否全局 rank 0（跨节点唯一的 0 号进程）
    print_on_global_main,  # 仅全局 rank 0 打印（全局唯一日志）
    print_on_local_main,  # 仅本节点 0 号卡打印（每节点一条）
)
from deepspec.trainer.ckpt_manager import (  # checkpoint 保存/发现/恢复
    discover_latest_checkpoint,  # 解析 step_latest 符号链接找最近 checkpoint
    load_resume_draft_model,  # 从 checkpoint 恢复 draft 模型权重
    load_training_state,  # 恢复优化器、进度计数与四路 RNG 状态
    save_checkpoint,  # 保存模型权重 + 各 rank 训练状态 + 原子更新 step_latest
)
import deepspec.utils.training_logger as training_logger  # step 级指标日志（TensorBoard）
from deepspec.utils.hfai_suspend import SuspendController  # 平台级"暂停-续跑"控制器（断点保活）


_PRECISION_DTYPES = {  # 精度名 → torch dtype：draft 训练权重/计算精度
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_SHARDING_STRATEGIES = {  # 配置名 → FSDP 分片策略；hybrid 系在节点内分片、节点间复制
    "full_shard": ShardingStrategy.FULL_SHARD,  # 参数/梯度/优化器状态全分片（类 ZeRO-3）
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,  # 仅梯度+优化器状态分片（类 ZeRO-2）
    "no_shard": ShardingStrategy.NO_SHARD,  # 不分片（类 DDP，小模型调试用）
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,  # 节点内全分片 + 节点间复制：通信量与显存折中
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,  # 混合 ZeRO-2 变体（私有枚举别名）
    "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,  # 同上的下划线别名（兼容旧配置）
}

_HYBRID_STRATEGIES = (  # 需要二维 device_mesh 的策略集合：决定下面是否建 replicate/shard 网格
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)


def _build_fsdp_kwargs(
    *, sharding_strategy_name: str, precision_dtype, world_size: int
) -> dict:
    # 组装 FSDP 构造参数：分片策略 + 混合精度；混合分片策略额外构造二维设备网格。
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    fsdp_kwargs = dict(
        use_orig_params=True,  # 保留原始参数名/形状：state_dict 与 HF save_pretrained 直接兼容
        mixed_precision=MixedPrecision(  # 混合精度：计算/缓冲用低精度，主权重保持高精度
            param_dtype=precision_dtype,  # 前向/反向时的参数计算精度
            buffer_dtype=precision_dtype,  # buffer（如 LayerNorm 的 running 量）精度
        ),
        sharding_strategy=sharding_strategy,
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        devices_per_node = torch.cuda.device_count()  # 每节点卡数：决定网格第二维大小
        fsdp_kwargs["device_mesh"] = init_device_mesh(  # 二维网格：replicate（跨节点复制）× shard（节点内分片）
            "cuda",
            (world_size // devices_per_node, devices_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    return fsdp_kwargs


def _compute_gradient_accumulation_steps(
    *, world_size: int, local_batch_size: int, global_batch_size: int
) -> int:
    # 梯度累积步数 = 全局批 / (每卡批 × 卡数)：每多少个 micro-batch 才做一次 optimizer.step。
    # 必须整除，否则全局批大小语义不成立（最后一个不完整累积窗口会破坏分布式梯度一致性）。
    denom = world_size * local_batch_size
    assert global_batch_size % denom == 0, (
        "global_batch_size must be divisible by world_size * local_batch_size: "
        f"global_batch_size={global_batch_size}, world_size={world_size}, "
        f"local_batch_size={local_batch_size}"
    )
    return global_batch_size // denom


def _compute_samples_per_epoch(*, dataset_size: int, global_batch_size: int) -> int:
    # 一个 epoch 实际使用的样本数：向下取整到全局批的整数倍，丢弃不足以凑满一批的尾巴。
    # 尾样本若参与训练会造成各 rank 批量不齐/部分 rank 空转，破坏 FSDP 梯度同步。
    samples_per_epoch = (dataset_size // global_batch_size) * global_batch_size
    assert samples_per_epoch > 0, (
        "train dataset is too small to form one full global batch: "
        f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
    )
    return samples_per_epoch


def _compute_training_schedule(
    *,
    world_size: int,
    dataset_size: int,
    local_batch_size: int,
    global_batch_size: int,
    num_train_epochs: int,
    max_train_steps=None,
) -> tuple[int, int, int, int, int, int, int]:
    # 派生完整训练日程：给定"epoch 数"或"总步数"二者之一，解出全部中间量。
    # max_train_steps 优先：指定时反推需要几个 epoch（向上取整保证能跑满步数）。
    gradient_accumulation_steps = _compute_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
    )
    samples_per_epoch = _compute_samples_per_epoch(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
    )
    per_rank_samples_per_epoch = samples_per_epoch // world_size  # 每 rank 每 epoch 消费样本数
    micro_batches_per_epoch = per_rank_samples_per_epoch // local_batch_size  # 每 rank 每 epoch micro-batch 数
    steps_per_epoch = micro_batches_per_epoch // gradient_accumulation_steps  # optimizer.step 次数/epoch
    if max_train_steps is None:
        resolved_max_train_steps = int(num_train_epochs) * steps_per_epoch  # 按 epoch 数推总步数
        resolved_num_train_epochs = int(num_train_epochs)
    else:
        resolved_max_train_steps = int(max_train_steps)  # 指定总步数：epoch 数反推
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / steps_per_epoch
        )
    return (
        gradient_accumulation_steps,
        samples_per_epoch,
        per_rank_samples_per_epoch,
        micro_batches_per_epoch,
        steps_per_epoch,
        resolved_max_train_steps,
        resolved_num_train_epochs,
    )


def _launch_eval(
    *,
    target_model_name_or_path: str,  # target 模型标识（评估配置用）
    checkpoint_dir: str,  # 刚保存的 checkpoint 目录（内含 draft 模型权重）
    step: int,  # 当前全局步数（评估报告标注用）
    tensorboard_dir: str,  # 评估结果回写的 TensorBoard 目录
    exp_name: str,  # 实验名
) -> None:
    # 保存 checkpoint 后自动提交评估任务（仅全局 rank0 调用）：
    # 若配置了 auto_eval_command 则拼命令行用 os.system 投递（不阻塞训练进程自身）。
    from deepspec.utils.constant import auto_eval_command  # 延迟导入：允许用户在 constant 中自定义
    if auto_eval_command is not None:
        command = auto_eval_command(target_model_name_or_path,checkpoint_dir,step,tensorboard_dir,exp_name)
        print_on_global_main(f"Submitting auto eval for {checkpoint_dir}")
        print_on_global_main(command)
        os.system(command)  # 同步提交（shell 返回即认为任务已投出），评估在别处异步执行
    else:
        print("You can use this function to launch your auto eval script!")

class BaseTrainer:
    # 训练器基类：负责与算法无关的全部工程——分布式初始化、FSDP 封装、目标缓存加载与校验、
    # 训练日程推导、优化器构建、梯度累积训练循环、checkpoint 保存/自动恢复/自动评估、
    # 以及平台级 suspend 续跑。子类（EAGLE3/DSpark）只需提供 _build_draft_model 与 run_batch。
    data_collator_cls = None  # 子类指定 collator 类（如 CacheCollator）；实例化时无参构造

    def __init__(self, local_rank, args):
        self.args = args  # 全量训练配置（OmegaConf 风格树形配置）
        self.device, self.global_rank, self.world_size = init_dist(local_rank)  # NCCL 进程组 + 本卡设备
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]  # 训练精度 dtype
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir  # checkpoint 根目录
        self.resume_checkpoint_dir = discover_latest_checkpoint(  # 自动续训入口：step_latest 指向的最新存档
            self.checkpoint_dir_root
        )
        self.suspend_controller = SuspendController(device=self.device)  # 平台暂停信号监听器
        self.next_micro_step = 0  # 进度的唯一真值：已完成的 micro-batch 数（恢复后由 checkpoint 覆盖）

        if is_global_main_process(): ensure_dir(self.checkpoint_dir_root)  # 仅 rank0 建目录，避免并发竞争
        training_logger.init(  # 初始化 step 级日志/TensorBoard 写手
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
        )

        self.draft_model, self.tokenizer = self.build_models()  # 构建 draft 模型并从 target 继承嵌入/头
        if self.resume_checkpoint_dir is not None:  # 有存档：先覆盖为 checkpoint 权重再上 FSDP
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
            )
        self.model = self.draft_model  # 训练视角的模型引用（后续会被 compile/FSDP 替换）
        if self.args.train.torch_compile:
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)  # dynamic=True：变长序列不再反复重编译
        self.model = self._wrap_with_fsdp(self.model)  # FSDP 封装：参数分片 + 混合精度

        self.train_dataset = CacheDataset(cache_dir=self.args.data.target_cache_path)  # 目标缓存数据集
        validate_train_cache(  # 缓存 ↔ draft 结构 ↔ target 模型名三方对齐校验
            train_dataset=self.train_dataset,
            draft_model=self.draft_model,
            target_model_name_or_path=self.args.model.target_model_name_or_path,
        )

        (
            self.gradient_accumulation_steps,  # 每次 optimizer.step 需要的 micro-batch 数
            self.samples_per_epoch,  # 每 epoch 全局消费样本数（对齐全局批）
            self.per_rank_samples_per_epoch,  # 每 rank 每 epoch 样本数
            self.micro_batches_per_epoch,  # 每 rank 每 epoch micro-batch 数
            self.steps_per_epoch,  # 每 epoch optimizer.step 数
            self.max_train_steps,  # 总 optimizer.step 数
            self.args.train.num_train_epochs,  # epoch 数（可能被 max_train_steps 反推覆盖）
        ) = _compute_training_schedule(
            world_size=self.world_size,
            dataset_size=len(self.train_dataset),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

        self.optimizer = BF16Optimizer(  # AdamW + warmup/余弦调度，内部按 BF16 主权重管理状态
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=self.max_train_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        if self.resume_checkpoint_dir is not None:  # 续训：恢复优化器状态、进度与四路 RNG
            resume_state = load_training_state(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                optimizer=self.optimizer,
                global_rank=self.global_rank,
                world_size=self.world_size,
                local_batch_size=int(self.args.train.local_batch_size),
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                micro_batches_per_epoch=self.micro_batches_per_epoch,
            )
            self.next_micro_step = resume_state.next_micro_step  # 从存档处继续计数
        else:
            print_on_local_main("Training from scratch.")
        self.info_board()  # 打印训练日程总览

    @property
    def global_step(self):
        # 全局步 = 已完成 micro 步数 ÷ 累积步数（整除语义由保存/恢复的对齐断言保证）。
        return self.next_micro_step // self.gradient_accumulation_steps

    def info_board(self):
        # 打印训练关键量：数据规模、批大小、累积步数、每 epoch 步数与总步数。
        print_on_local_main("***** Running training *****")
        print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(f"  Global batch size = {self.args.train.global_batch_size}")
        print_on_local_main(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")

    def build_models(self):
        # 构建 draft 模型 + 从 target 模型继承冻结的嵌入层与 lm_head。
        # 只为拿 embed/lm_head 才完整加载 target（CPU + eval，随后即删）——
        # draft 的主体参数是新初始化的，target 权重仅在初始化时被拷贝。
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(  # 词表与 chat template 均取自 target
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(  # 用 target 结构推导 draft 结构（层数/维度等）
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(  # 子类钩子：按算法家族构建 draft 模型
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # Training only uses the target checkpoint to initialize frozen draft
        # embeddings and lm_head weights.
        target_model = AutoModelForCausalLM.from_pretrained(  # CPU 加载完整 target 仅为取嵌入/头
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()  # 输入嵌入矩阵
        target_lm_head = target_model.get_output_embeddings()  # 输出投影（lm_head）
        assert (target_lm_head is not None) and (target_embed_tokens is not None)
        draft_model.initialize_embeddings_and_head(  # 拷贝进 draft 并冻结：draft 不重学词表
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model  # 立即释放 target 占用的内存
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        # 子类必须实现的钩子：按算法（EAGLE3/DSpark）与模型家族构建 draft 模型实例。
        raise NotImplementedError

    def _wrap_with_fsdp(self, model):
        # 用配置指定的分片策略与精度把模型包成 FSDP（各 rank 参数分片持有）。
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            world_size=self.world_size,
        )
        return FSDP(model, **fsdp_kwargs)

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        # 构建训练 DataLoader：sampler 支持从"全局样本偏移"续训，跳过已消费样本。
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            total_size=self.samples_per_epoch,  # 对齐全局批后的 epoch 样本数
            start_global_offset_samples=start_offset_samples,  # 续训起点：跳过已训练样本
            num_samples=num_samples,  # 本次实际要取的样本数（剩余量）
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator_cls(),  # 子类指定的 collator（CacheCollator 等）
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,  # 锁页内存：配合 CUDA 预取器的 non_blocking H2D 才能真正异步
            drop_last=True,  # 丢尾批：保证各 rank 批量一致且 FSDP 梯度同步语义正确
            persistent_workers=True,  # worker 进程跨 epoch 存活：避免每轮重建进程/缓存
            prefetch_factor=4,  # 每 worker 预取批数：CPU 侧也保持流水不断
        )

    def run_batch(self, batch):
        # 子类必须实现：单 micro-batch 前向 + 损失计算（返回标量 loss 张量）。
        raise NotImplementedError

    def _checkpoint_kwargs(self):
        # 汇总 save_checkpoint 所需的全部上下文（模型/优化器/进度/分布式拓扑）。
        return dict(
            model=self.model,  # FSDP（可能已 compile）封装后的训练模型
            draft_model=self.draft_model,  # 未封装的 draft：提供 save_pretrained 元信息
            optimizer=self.optimizer,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,  # 进度真值
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
        )

    def save_and_eval_checkpoint(self):
        # 保存 checkpoint 并由全局 rank0 提交自动评估；barrier 保证所有 rank 存完才继续。
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
        if is_global_main_process():
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
            )
        dist.barrier()  # 等 rank0 完成保存+提交，防止其他 rank 提前进入下一窗口产生竞态
        return checkpoint_dir

    def _save_and_suspend(self):
        # 平台要求暂停（如集群抢占/错峰）：先完整落盘 checkpoint，再调用平台挂起接口。
        # 挂起发生在 optimizer.step 边界，恢复时从 step_latest 精确续跑。
        print_on_global_main("Saving checkpoint before suspending...")
        save_checkpoint(**self._checkpoint_kwargs())
        dist.barrier()  # 所有 rank 落盘完成
        if is_global_main_process():
            print_on_global_main("Going to suspend...")
            self.suspend_controller.go_suspend()  # 触发平台级挂起（进程被冻结/调度走）
        dist.barrier()  # 挂起返回（续跑）后的汇合点

    def train(self):
        # 主训练循环：按 micro-batch 前向/反向，每 gradient_accumulation_steps 步做一次
        # 裁剪梯度 + optimizer.step + 日志 + 周期性 checkpoint/评估 + suspend 检查。
        # 断点续训通过"样本偏移起点"实现：dataloader 只从 next_micro_step 对应的样本开始拉。
        self.model.train()  # FSDP/compile 包装模型进入训练模式
        if self.global_step >= self.max_train_steps:
            return  # 恢复后已达标：无需再训

        local_batch_size = int(self.args.train.local_batch_size)
        total_micro_steps = self.max_train_steps * self.gradient_accumulation_steps  # 全程 micro 步总数
        remaining_micro_steps = total_micro_steps - self.next_micro_step  # 剩余 micro 步
        remaining_samples = remaining_micro_steps * local_batch_size  # 剩余样本数（本 rank 视角）

        dataloader = self._build_train_dataloader(  # 从续训偏移开始构建本 epoch 剩余数据流
            start_offset_samples=self.next_micro_step * local_batch_size,
            num_samples=remaining_samples,
        )
        prefetcher = CUDAPrefetcher(dataloader, self.device)  # H2D 预取与计算重叠
        training_logger.start_session(global_step=self.global_step)

        with self.suspend_controller.monitoring():  # 上下文内轮询平台暂停请求
            for batch in prefetcher:
                should_sync = (  # 是否到达梯度同步点（累积窗口末尾）
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else self.model.no_sync()  # 非同步窗口内禁用梯度 all-reduce
                with sync_context:
                    loss = self.run_batch(batch) / self.gradient_accumulation_steps  # 损失按累积步数缩放：等效全局批平均
                    loss.backward()
                self.next_micro_step += 1  # micro 步计数推进（唯一进度真值）

                if not should_sync:
                    continue  # 累积中：跳过同步与更新

                grad_norm = FSDP.clip_grad_norm_(  # FSDP 感知的梯度裁剪：先 all-reduce 范数再统一缩放
                    self.model,
                    float(self.args.train.max_grad_norm),
                )
                self.optimizer.step()  # 真正的参数更新（内部含 LR 调度推进）
                training_logger.on_optimizer_step(  # step 级日志：loss/lr/grad_norm 等
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self.optimizer.get_learning_rate(),
                    grad_norm=grad_norm.item(),
                )

                if self.global_step % int(self.args.logging.checkpointing_steps) == 0:
                    self.save_and_eval_checkpoint()  # 周期保存 + 自动评估

                if self.suspend_controller.requested():
                    self._save_and_suspend()  # 平台暂停：落盘并挂起（续跑后 return 结束本轮）
                    return

        self.save_and_eval_checkpoint()  # 训练自然结束：保存最终 checkpoint

    def clean_up(self):
        # 退出清理：关日志 → 全员到齐 → 销毁进程组（顺序保证不丢单 rank 的收尾日志）。
        training_logger.close()
        dist.barrier()
        dist.destroy_process_group()
