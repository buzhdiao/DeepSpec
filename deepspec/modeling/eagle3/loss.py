"""Triton-fused soft cross-entropy for Eagle3 TTT distillation.

Ports SpecForge's ``LogSoftmaxLoss`` from
``SpecForge/specforge/core/loss.py`` (Apache-2.0, itself incorporating
Unsloth and Liger-Kernel ideas). See the NOTICE file at the repository
root for third-party attribution. Masked positions write zero loss, and
``compute_eagle3_loss`` supplies the configured normalizer for each TTT step.

The fused forward/backward keeps only ``logits.detach()`` + two fp32
per-row scalars (m, d) across autograd's save_for_backward, never
materialising the [B, T, V] fp32 log-probs tensor that the naive
PyTorch path retains across TTT steps.
"""

# 本模块实现 Eagle3 TTT（训练时模拟多步起草）蒸馏损失：
#  - FusedLogSoftmaxLoss：Triton 融合的 soft 交叉熵（前向+反向），反向梯度原地写回 logits 存储；
#  - compute_eagle3_loss：draft 以整条序列为块自回归滚动 ttt_length 步，
#    每步对目标模型的下一 token 分布做 soft 交叉熵并按步衰减加权求和。
import torch  # PyTorch：张量、autograd 与设备管理
import triton  # Triton：GPU 核函数编译与启动
import triton.language as tl  # Triton 语言原语：块级加载/计算/存储
from transformers import DynamicCache  # HF 动态 KV cache：TTT 各步的 draft 缓存按块追加

from deepspec.utils.metrics import add_metric  # 训练指标记录（窗口内累加，统一 flush 聚合）


def _calculate_settings(n: int):
    # 依据词表大小选取 Triton kernel 的块大小与 warp 数。
    # BLOCK_SIZE is the per-iteration chunk; the kernel loops over V in
    # chunks of BLOCK_SIZE, so large vocabularies (e.g. Qwen3 151936) do
    # NOT need BLOCK_SIZE == next_pow2(V). Cap at MAX_FUSED_SIZE to keep
    # register / shared memory usage in bounds.
    MAX_FUSED_SIZE = 131072  # 单块上限（2^17）：寄存器/共享内存预算硬顶
    BLOCK_SIZE = min(triton.next_power_of_2(n), MAX_FUSED_SIZE)  # 词表 2 的幂上取整，再截到上限

    num_warps = 4  # 小块默认 4 warp
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    if hasattr(torch.version, "hip") and torch.version.hip is not None:  # AMD ROCm (HIP) 后端
        num_warps //= 2  # HIP 上每 CU 资源更紧张，warp 数减半

    return BLOCK_SIZE, num_warps


@torch.no_grad()
def _shift_with_zero_padding(
    tensor: torch.Tensor,  # 输入 [batch, seq]，沿时间维移动
    *,
    left: bool = True,  # True 右移（头部补零）；False 左移（尾部补零）
) -> torch.Tensor:
    # 沿时间维移动一格并以 0 填补空位；TTT 每步用它把 draft 输入窗整体左移一位。
    zero_padding = torch.zeros_like(tensor[:, -1:])  # 与最后一列同形状的零块 [batch, 1]
    if left:
        return torch.cat((zero_padding, tensor[:, :-1]), dim=1)  # 右移：丢末列、头部补零
    return torch.cat((tensor[:, 1:], zero_padding), dim=1)  # 左移：丢首列、尾部补零


def _compute_loss_normalizers(
    *,
    position_masks: list[torch.Tensor],  # 各 TTT 步的位置掩码 [batch, seq, 1]
    loss_normalization: str = "local_mean",  # 归一化方式，Eagle3 仅允许 local_mean
) -> list[float]:
    # 计算每个 TTT 步的损失归一子：每步除以本地 B*T，使每条序列等权。
    # Eagle3 only supports local_mean: each TTT step divides by the local
    # B*T (per-sequence at local_batch_size=1), which weights every sequence
    # equally. valid_token_mean is intentionally disabled because its global
    # token-count divisor weights all tokens equally, so long sequences
    # dominate the gradient and eval accept length degrades over training.
    assert loss_normalization != "valid_token_mean", (
        "valid_token_mean normalization is disabled; Eagle3 uses local_mean."
    )
    return [
        float(int(position_mask.shape[0]) * int(position_mask.shape[1]))  # B*T（与掩码内容无关，恒为本地 token 总数）
        for position_mask in position_masks
    ]


@torch.no_grad()
def _build_next_token_position_mask(
    *,
    loss_mask: torch.Tensor,  # 逐 token 监督掩码 [batch, seq]（1=有监督）
    attention_mask: torch.Tensor,  # padding 掩码 [batch, seq]
) -> torch.Tensor:
    # Last valid tokens have no next-token target inside the cached sequence.
    # 构造 step 0 的"下一 token"位置掩码 [batch, seq, 1]：每行最后一个有效 token
    # 在本窗口内没有下一 token 目标，置 0 屏蔽；其余位置沿用 loss_mask。
    position_mask = loss_mask.to(torch.float32).clone()  # 拷贝为 fp32，避免改写原掩码
    seq_lengths = attention_mask.to(torch.long).sum(dim=-1)  # 每行有效长度 [batch]
    batch_indices = torch.arange(position_mask.shape[0], device=position_mask.device)  # 行号 0..B-1
    last_token_indices = (seq_lengths - 1).clamp_min(0)  # 每行最后一个有效位置（空行退化为 0）
    position_mask[batch_indices, last_token_indices] = 0  # 屏蔽末位有效 token
    return position_mask.unsqueeze(-1)  # [batch, seq, 1]，与 logits 前两维对齐


@torch.no_grad()
def _build_padded_next_token_target_probs(
    target_logits: torch.Tensor,  # 目标侧词表 logits [batch, seq, vocab]
    ttt_length: int,  # TTT 步数
) -> torch.Tensor:
    # The extra uniform tail keeps each TTT slice the same length.
    # 生成按步切片的"下一 token"目标分布 [batch, seq+ttt_length+1, vocab]：
    # 丢掉第 0 位后，结果第 j 位 = 原 token j+1 之后的 token 分布——与 draft
    # 第 s 步槽位 i（输入为原 token i+s+1）的监督目标严格对齐；尾部补
    # ttt_length+1 行均匀分布，使越界切片也满长（其损失由位置掩码清零）。
    target_probs = torch.softmax(target_logits.float(), dim=-1)  # fp32 softmax，数值稳定
    batch_size, _, vocab_size = target_probs.shape
    uniform_tail = target_probs.new_full(
        (batch_size, int(ttt_length) + 1, vocab_size),
        1.0 / float(vocab_size),  # 均匀分布 1/V
    )
    return torch.cat((target_probs[:, 1:, :], uniform_tail), dim=1).detach()  # 左移一位拼接均匀尾巴；蒸馏目标不回传梯度


@torch.no_grad()
def _log_eagle3_step_metrics(
    *,
    step_idx: int,  # 当前 TTT 步序号（用于指标命名）
    draft_logits: torch.Tensor,  # draft 词表 logits [batch, seq, vocab]
    target_probs: torch.Tensor,  # 对齐后的目标分布 [batch, seq, vocab]
    position_mask: torch.Tensor,  # 本步位置掩码 [batch, seq, 1]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 记录单个 TTT 步的训练指标：top-1 命中率（accuracy）、以 1-TV/2 近似的
    # 拒绝采样接受率（accept_rate）、有效 token 数；并返回三组掩码供前缀指标聚合。
    valid_mask = position_mask.squeeze(-1) > 0  # 本步有监督的位置 [batch, seq]
    correct_mask = (draft_logits.argmax(-1) == target_probs.argmax(-1)) & valid_mask  # top-1 一致且有效
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)  # draft 概率
    accept_rate_mask = 1.0 - 0.5 * (
        draft_probs - target_probs.float()
    ).abs().sum(dim=-1)  # 1 - TV/2：总变差距离的一半，即拒绝采样下单 token 期望接受率的界
    accept_rate_mask = accept_rate_mask.clamp_(0.0, 1.0)  # 数值上截断到 [0, 1]
    accept_rate_mask = accept_rate_mask * valid_mask.to(torch.float32)  # 无效位置清零

    correct = correct_mask.to(torch.float32).sum()  # 命中数（ratio 分子）
    valid_count = valid_mask.to(torch.float32).sum()  # 有效数（ratio 分母）
    accept_sum = accept_rate_mask.sum()  # 接受率总和（ratio 分子）
    add_metric(f"accuracy@{step_idx}", correct, den=valid_count, tag="train")  # 全局按 sum(num)/sum(den) 聚合
    add_metric(f"accept_rate@{step_idx}", accept_sum, den=valid_count, tag="train")
    add_metric(
        f"valid_tokens@{step_idx}",
        valid_count,
        reduction="dp_sum",  # 数据并行各 rank 直接求和
        tag="train",
    )
    return correct_mask, accept_rate_mask, valid_mask


@torch.no_grad()
def _log_eagle3_prefix_metrics(
    *,
    correct_masks: list[torch.Tensor],  # 各步 top-1 命中掩码
    accept_rate_masks: list[torch.Tensor],  # 各步接受率掩码
    valid_masks: list[torch.Tensor],  # 各步有效掩码
) -> None:
    # 汇总整个 TTT rollout 的前缀指标：估计投机解码的期望验收长度 τ。
    # greedy：命中掩码沿步维 cumprod = "此前各步 draft 全部命中"，即贪心起草下
    # 该位置已连续被接受的 draft token 数；probabilistic：接受率掩码 cumprod 的
    # 期望版本。两者每位置再 +1（乘 step 0 有效掩码后求和），对应目标模型必中的
    # bonus token，即 τ = E[连续接受 draft 数] + 1。
    correct_tensor = torch.stack(
        [correct_mask.to(torch.float32) for correct_mask in correct_masks],
        dim=0,  # [ttt_length, batch, seq]
    )
    start_valid = valid_masks[0].to(torch.float32)  # 以 step 0 的有效位置为统计基准
    tau_count = start_valid.sum()  # 基准有效 token 数（ratio 分母）
    accepted_draft_tokens = correct_tensor.cumprod(dim=0).sum(dim=0)  # 各位置"前缀全命中"的步数
    tau_greedy_sum = (accepted_draft_tokens * start_valid).sum() + tau_count  # 每位置 +1 后求和（ratio 分子）

    accept_rate_tensor = torch.stack(accept_rate_masks, dim=0).to(torch.float32)
    expected_draft_tokens = accept_rate_tensor.cumprod(dim=0).sum(dim=0)  # 期望连续接受 draft token 数
    tau_prob_sum = (expected_draft_tokens * start_valid).sum() + tau_count

    add_metric("tau_greedy", tau_greedy_sum, den=tau_count, tag="train")  # 全局 ratio：sum/sum
    add_metric("tau_probabilistic", tau_prob_sum, den=tau_count, tag="train")


@triton.jit
def _log_softmax_forward_kernel(
    logits_ptr,  # logits 指针：每行（一个 (b,t) 位置）一个 program
    logits_stride,  # logits 行步长（元素数）
    target_ptr,  # soft 标签指针（与 logits 同形状 [rows, V]）
    target_stride,  # soft 标签行步长
    position_mask_ptr,  # 位置掩码指针 [rows]
    position_mask_stride,  # 掩码行步长
    loss_ptr,  # 输出：逐行损失 [rows, 1]
    loss_stride,  # 损失行步长
    m_ptr,  # 输出：每行 logsumexp 的最大值 m（fp32），backward 复用
    d_ptr,  # 输出：每行 exp 和 d（fp32），backward 复用
    n_cols,  # 词表大小 V
    BLOCK_SIZE: tl.constexpr,  # 每次迭代处理的词表块大小
):
    # 融合的 soft 交叉熵前向：两遍扫描在线计算 logsumexp——第一遍求运行最大值 m
    # 与运行 exp 和 d，第二遍累加 -Σ_v target_p[v]·(logits-m-log d)；全程不物化
    # [B*T, V] 的 fp32 log-probs。掩码为 0 的行直接返回（损失占位保持 0）。
    program_id = tl.program_id(0).to(tl.int64)  # 行号；int64 防大张量地址溢出
    logits_ptr += program_id * logits_stride  # 定位到本行
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id * position_mask_stride
    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        return  # 被屏蔽的行不计算，loss 缓冲保持初始化的 0

    m = float("-inf")  # 运行最大值
    d = 0.0  # 运行归一子和 Σ exp(logits - m)

    for i in range(0, n_cols, BLOCK_SIZE):  # 第一遍：在线 logsumexp
        offsets = i + tl.arange(0, BLOCK_SIZE)  # 本块列偏移
        mask = offsets < n_cols  # 越界列屏蔽
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=float("-inf")
        ).cast(tl.float32)  # fp32 计算，保证数值稳定
        block_max = tl.max(tl.where(mask, logits_block, float("-inf")))  # 块内最大
        m_new = tl.maximum(m, block_max)  # 合并历史最大值
        d = d * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(logits_block - m_new), 0.0)
        )  # 旧和按新最大值重缩放，再加本块贡献
        m = m_new

    loss = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):  # 第二遍：累加 -Σ target_p · log_softmax
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        normalized_logits = logits_block - m  # 减最大值防溢出
        log_normalizer = tl.log(d)  # log Σ exp
        log_softmax_logits = normalized_logits - log_normalizer  # log_softmax(logits)
        weighted_log_prob = target_block * log_softmax_logits  # soft 标签加权
        loss += tl.sum(tl.where(mask, weighted_log_prob, 0.0))

    loss_ptr += program_id * loss_stride
    m_ptr += program_id
    d_ptr += program_id
    tl.store(loss_ptr, -loss)  # 交叉熵取负号
    tl.store(m_ptr, m.to(tl.float32))  # 保存 logsumexp 统计量，backward 免重扫
    tl.store(d_ptr, d.to(tl.float32))


@triton.jit
def _log_softmax_backward_kernel(
    logits_ptr,  # logits 指针（同时作为梯度写出缓冲）
    logits_stride,
    target_ptr,  # soft 标签指针
    target_stride,
    position_mask_ptr,  # 位置掩码指针 [rows]
    grad_output_ptr,  # 上游标量梯度（对所有行共享）
    scaling_factor,  # 1/normalizer：把行和损失缩放成均值
    m_ptr,  # forward 存下的每行最大值 m
    d_ptr,  # forward 存下的每行 exp 和 d
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 融合反向：d loss_row / d logits = -(target_p - softmax(logits))，再乘上游梯度
    # 与 1/normalizer；梯度原地写回 logits 存储（Liger/Unsloth 省显存模式，调用方
    # 此后不得再读 logits）。掩码为 0 的行写全 0 梯度。
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id

    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:  # 无监督行：梯度整行清零
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(logits_ptr + offsets, 0.0, mask=mask)
        return

    m_ptr += program_id
    d_ptr += program_id
    m = tl.load(m_ptr).to(tl.float32)  # 复用前向统计量，免二次扫描求 max/sum
    d = tl.load(d_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)  # 对总损失标量的上游梯度
    grad_output = grad_output * scaling_factor  # sum -> mean 缩放

    target_grad_sum = 0.0  # Σ_v target_p[v]·g：softmax 项的公共系数
    for i in range(0, n_cols, BLOCK_SIZE):  # 第一遍：预累加公共系数
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_grad_sum += tl.sum(tl.where(mask, target_block * grad_output, 0.0))

    for i in range(0, n_cols, BLOCK_SIZE):  # 第二遍：写回 -(target_p·g - softmax·Σtarget_p·g)
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        softmax_prob = tl.exp(logits_block - m) / d  # softmax(logits)
        normalized_grad = softmax_prob * target_grad_sum  # softmax 项
        grad_block = -(target_block * grad_output - normalized_grad)
        tl.store(logits_ptr + offsets, grad_block.to(tl.float32), mask=mask)


# 融合 Triton 前向/反向的 soft 交叉熵 autograd.Function。
# save_for_backward 只保留 logits.detach() + 目标分布 + 掩码 + 两个 fp32 行标量
# （m, d），TTT 各步间不再持有 [B, T, V] fp32 log-probs；反向把梯度原地写进
# detached 的 logits 存储——此后调用方不得再读 logits。
class FusedLogSoftmaxLoss(torch.autograd.Function):
    """Soft cross-entropy with fused Triton forward/backward.

    Returns ``sum_i(-sum_v target_p[i,v] * log_softmax(logits)[i,v]) /
    normalizer``, where masked positions contribute zero to the numerator.

    Backward writes the gradient in-place into ``logits`` storage via a
    detached view and returns that tensor as the gradient. This is the
    Liger/Unsloth memory-saving pattern: callers must not read ``logits``
    after this loss has run backward.
    """

    @staticmethod
    def forward(ctx, logits, target_p, position_mask, normalizer):
        # 前向：拉平 [B,T,V] 后每行启动一个 program，返回行损失之和 / normalizer。
        assert logits.is_cuda, "FusedLogSoftmaxLoss requires CUDA tensors."  # Triton kernel 仅支持 CUDA
        assert logits.shape == target_p.shape  # logits 与 soft 标签逐元素对齐 [B, T, V]
        assert position_mask.shape[:2] == logits.shape[:2]  # 掩码前两维与 logits 对齐
        B, T, V = logits.shape
        loss = torch.zeros((B * T, 1), device=logits.device, dtype=torch.float32)  # 逐行损失输出缓冲（掩码行保持 0）
        logits_flat = logits.contiguous().view(B * T, V)  # 拉平为 [B*T, V]，kernel 按行处理
        target_flat = target_p.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)  # 每行一个 program
        m = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)  # 每行最大值输出
        d = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)  # 每行 exp 和输出
        BLOCK_SIZE, num_warps = _calculate_settings(V)  # 按词表大小选块与 warp 数
        _log_softmax_forward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),  # 行步长
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            V,  # n_cols = 词表大小
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(logits.detach(), target_p, position_mask, m, d)  # 只存 detached logits + 两个行标量，省显存
        ctx.normalizer = float(normalizer)  # 归一子挂在 ctx 上供 backward 用
        return loss.squeeze(1).sum() / ctx.normalizer  # 行损失求和后除以本地 B*T

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：kernel 把梯度原地写进 detached logits 存储，并将其作为 logits 梯度返回。
        logits, target, position_mask, m, d = ctx.saved_tensors
        B, T, V = logits.shape
        scaling_factor = 1.0 / ctx.normalizer  # sum -> mean 的缩放系数
        logits_flat = logits.contiguous().view(B * T, V)  # 梯度写回缓冲（即 detached logits 本体）
        target_flat = target.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        _log_softmax_backward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            grad_output,  # 上游标量梯度
            scaling_factor,
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return logits_flat.view(B, T, V), None, None, None  # 只有 logits 可导；target_p/掩码/归一子梯度为 None


def compute_eagle3_loss(
    *,
    model,  # Eagle3 draft 模型（Qwen3/Gemma4Eagle3Model）
    batch: dict[str, torch.Tensor],  # 训练批次：input_ids/attention_mask/loss_mask/target 隐状态等
    ttt_length: int,  # TTT 步数：一次训练迭代模拟的自回归起草步数
    step_loss_decay: float,  # 步间衰减系数：第 s 步损失权重为 decay^s
) -> torch.Tensor:
    # Eagle3 TTT 蒸馏损失主循环：draft 以整条序列为一块，共享自身 KV cache 自回归
    # 滚动 ttt_length 步——每步把上一步 draft 输出隐状态当作本步特征输入（模拟推理
    # 时多 token 起草），各步 draft logits 对目标模型"下一 token"分布做 soft 交叉熵，
    # 按 decay^s 加权累加；越深的步误差累积越大、权重越低。
    input_ids = batch["input_ids"].long()  # [B, T] 目标输入 token
    attention_mask = batch["attention_mask"].long()  # [B, T] padding 掩码
    hidden_states = batch["target_hidden_states"]  # [B, T, 5*H] 目标多层隐状态拼接（Eagle3 输入特征）
    target_last_hidden_states = batch["target_last_hidden_states"]  # [B, T, H] 目标末层隐状态（算目标 logits）
    seq_len = int(input_ids.shape[1])  # 每步喂入的块长 = 整条序列长
    base_position_ids = torch.arange(
        seq_len,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0).expand(input_ids.shape[0], -1)  # [B, T] 基准位置（跨步 RoPE 偏移由模型侧处理）
    current_input_ids = _shift_with_zero_padding(input_ids, left=False)  # 左移一位：draft 输入领先目标一个 token（预测下一词）
    shifted_position_mask = _build_next_token_position_mask(
        loss_mask=batch["loss_mask"],  # 逐 token 监督掩码
        attention_mask=attention_mask,
    )  # step 0 的"下一 token"位置掩码 [B, T, 1]
    past_key_values = DynamicCache()  # draft 自身 KV cache：TTT 各步按整块追加
    total_loss = hidden_states.new_zeros((), dtype=torch.float32)  # fp32 标量累加器
    with torch.no_grad():  # 目标分布是冻结的蒸馏信号，不回传梯度
        target_logits = model(
            target_last_hidden_states=target_last_hidden_states,
            target_logits_only=True,  # 只把目标末层隐状态过一遍 lm_head
        )
    target_probs = _build_padded_next_token_target_probs(
        target_logits=target_logits,
        ttt_length=int(ttt_length),
    )  # [B, T+ttt_length+1, V]，各步切片取用
    del target_logits  # softmax 后 logits 不再需要，及时释放
    position_masks = []  # 预生成各步位置掩码
    for _ in range(int(ttt_length)):
        position_masks.append(shifted_position_mask)
        shifted_position_mask = _shift_with_zero_padding(
            shifted_position_mask.squeeze(-1),
            left=False,
        ).unsqueeze(-1)  # 每步掩码同步左移一位：有效目标区随输入窗左移收缩
    loss_normalizers = _compute_loss_normalizers(
        position_masks=position_masks,
    )  # 每步归一子 = 本地 B*T

    correct_masks = []  # 各步命中掩码（前缀指标用）
    accept_rate_masks = []
    valid_masks = []
    for step_idx in range(int(ttt_length)):
        # Keep this slice alignment in sync with the Eagle3 reference.
        target_step_probs = target_probs[
            :,
            step_idx : step_idx + seq_len,
            :,
        ].contiguous()  # 第 s 步监督目标：位置 i 对应"原 token i+s+2 的分布" [B, T, V]
        position_mask_step = position_masks[step_idx]
        output = model(
            hidden_states=hidden_states,  # 上一步 draft 输出（首步为 fc 投影的目标特征）
            input_ids=current_input_ids,  # 本步左移后的输入 token
            attention_mask=attention_mask,
            position_ids=base_position_ids,
            past_key_values=past_key_values,  # 跨步共享 draft KV cache（TTT 核心机制）
            use_cache=True,
            return_logits=True,
            rope_cache_step_offset=True,  # RoPE 位置 +lck，对齐原序列绝对尺度
        )
        hidden_states = output.hidden_states  # draft 输出隐状态成为下一步特征输入（TTT 自回归链）
        correct_mask, accept_rate_mask, valid_mask = _log_eagle3_step_metrics(
            step_idx=step_idx,
            draft_logits=output.draft_logits,
            target_probs=target_step_probs,
            position_mask=position_mask_step,
        )
        correct_masks.append(correct_mask)
        accept_rate_masks.append(accept_rate_mask)
        valid_masks.append(valid_mask)
        step_loss = FusedLogSoftmaxLoss.apply(
            output.draft_logits,  # backward 后其存储被改写为梯度，勿再读
            target_step_probs,
            position_mask_step,
            loss_normalizers[step_idx],  # 本步归一子（本地 B*T）
        )
        add_metric(
            f"ploss_{step_idx}",  # 每步损失单独记录
            step_loss.detach(),
            reduction="dp_mean",
            tag="train",
        )
        step_weight = float(step_loss_decay) ** step_idx  # 第 s 步权重 decay^s
        total_loss = total_loss + step_loss * step_weight
        current_input_ids = _shift_with_zero_padding(current_input_ids, left=False)  # 输入窗再左移一位，进入下一步

    _log_eagle3_prefix_metrics(
        correct_masks=correct_masks,
        accept_rate_masks=accept_rate_masks,
        valid_masks=valid_masks,
    )  # 汇总 τ_greedy / τ_probabilistic
    add_metric("loss", total_loss.detach(), reduction="dp_mean", tag="train")
    return total_loss


__all__ = [
    "FusedLogSoftmaxLoss",
    "compute_eagle3_loss",
]
