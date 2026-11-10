# DSpark 训练损失：掩码交叉熵（主项）+ 可选 L1 分布蒸馏 + 置信度头 BCE，基于全局分母实现多卡一致的加权平均。
from typing import Optional

import torch  # PyTorch：张量操作、CUDA 设备属性与同步
import torch.distributed as dist  # 分布式原语：all_reduce 汇总各卡损失分母
import torch.nn.functional as F  # cross_entropy / binary_cross_entropy_with_logits 等无状态算子

from deepspec.utils.metrics import add_metric  # 指标记录：ratio 型指标传分子/分母，flush 时全局 sum(num)/sum(den)
from .common import DSparkForwardOutput  # 一次前向的产物容器（draft_logits/eval_mask/置信度预测等）


def _all_reduce_loss_denominators(
    loss_terms: dict[str, torch.Tensor],  # 本卡累计的各损失分子/分母
    *,
    world_size: int,  # 数据并行世界大小；==1 时跳过通信
) -> dict[str, torch.Tensor]:
    # 汇总（all_reduce SUM）各卡分母，让每张卡用同一"全局分母"计算损失：
    # DDP 会把各卡梯度取平均，本卡损失 = 本卡分子 / 全局分母 × world_size，求平均后恰好等于
    # 全局分子 / 全局分母的梯度——等价于全体 token 上的全局均值，多卡权重与单卡完全一致。
    denominators = {}
    for key in ("ce_loss_den", "l1_loss_den", "confidence_loss_den"):  # 三类损失各自的分母键
        tensor = loss_terms[key].detach().clone()  # 脱离计算图并复制，避免 all_reduce 原地改写影响梯度
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 跨卡求和得到全局分母（阻塞式集合通信）
        denominators[key] = tensor
    return denominators


def _build_loss_weight_mask(
    *,
    eval_mask: torch.Tensor,  # [batch, num_blocks, block_size] 逐位监督掩码
    block_size: int,
    device: torch.device,
    loss_decay_gamma: Optional[float],  # 指数衰减时间常数 γ；None 或 <=0 表示不衰减
) -> torch.Tensor:
    # 构造逐位损失权重：eval_mask 之上可选乘 exp(-k/γ)（k 为块内偏移）。
    # 块内越靠后误差复利越重（前位错则后位上文错），γ 调节对远端草稿位的惩罚强度。
    loss_weight_mask = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size, device=device).view(1, 1, -1)  # [1, 1, block_size] 块内偏移 k
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))  # exp(-k/γ)：k=0 权重 1，k 越大越小
        loss_weight_mask = loss_weight_mask * decay_weights  # 广播相乘得逐位权重
    return loss_weight_mask


def _compute_local_probabilistic_stats(
    *,
    outputs: DSparkForwardOutput,
    accept_rate_3d: Optional[torch.Tensor],  # [batch, num_blocks, block_size] 逐位接受率估计 α
    valid_block_weights: torch.Tensor,  # [batch, num_blocks] 有效块权重（0/1）
) -> tuple[torch.Tensor, torch.Tensor]:
    # 计算概率口径的投机解码指标（在 no_grad 下调用，仅监控）：
    # 按接受链公式，第 k+1 个草稿 token 被接受要求前 k+1 位全部被接受，故期望接受草稿数
    # E[#accepted] = Σ_k Π_{j<=k} α_j；再 +1 计入草稿被采纳后必然回补的 target 修正 token，
    # 得到每块期望产出 token 数 τ（"tau_probabilistic" 指标）。另输出逐位 α 之和供按位置分桶统计。
    _, _, block_size, _ = outputs.draft_logits.shape
    device = outputs.draft_logits.device
    pos_accept_sums = torch.zeros(block_size, device=device, dtype=torch.float32)  # [block_size] 逐位 α 求和容器
    if accept_rate_3d is None:
        return outputs.draft_logits.new_zeros((), dtype=torch.float32), pos_accept_sums  # 无目标 logits 时返回 0 占位

    valid_accept_rate = accept_rate_3d * outputs.eval_mask.to(torch.float32)  # 无效位置 α 置 0
    pos_accept_sums = valid_accept_rate.sum(dim=(0, 1))  # 对批与块求和 → 逐位 α 总量（ratio 分子）
    expected_draft_accepted = valid_accept_rate.cumprod(dim=-1).sum(dim=-1)  # Σ_k Π_{j<=k} α_j：每块期望接受的草稿数
    tau_prob_per_block = expected_draft_accepted + 1.0  # +1：被采纳草稿后总能额外接受一个 target token
    tau_prob_sum = (tau_prob_per_block * valid_block_weights).sum()  # 仅对有效块加权求和（ratio 分子）
    return tau_prob_sum, pos_accept_sums


def _compute_accept_rate_3d(
    *,
    outputs: DSparkForwardOutput,
    aligned_target_logits: Optional[torch.Tensor],  # 目标模型在对应预测位置的 logits；None 表示未提供
) -> Optional[torch.Tensor]:
    # 逐位接受率估计 α = 1 - TV/2，其中 TV = Σ_v |p_draft(v) - p_target(v)| 是两分布的总变差距离。
    # 这正是拒绝采样下草稿 token 被目标接受的期望概率（分布重叠系数），作为置信度头的监督目标。
    if aligned_target_logits is None:
        return None
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)  # fp32 softmax 避免半精度数值误差
    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
    accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)  # 1 - TV/2 ∈ [0, 1]
    return accept_rate_3d.clamp_(0.0, 1.0)  # 原地钳位消除浮点误差越界


def _compute_local_l1_term(
    *,
    outputs: DSparkForwardOutput,
    aligned_target_logits: Optional[torch.Tensor],
    loss_weight_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # L1 蒸馏项：逐位 Σ_v |p_draft - p_target| 按 loss_weight_mask 加权，返回分子/分母
    # （全局分母归一在 all_reduce 之后完成）。相比 CE 只拉正类 logit，L1 直接约束整条分布
    # 向目标分布对齐，从而给置信度头提供更准的 p_draft。
    zero = outputs.draft_logits.new_zeros((), dtype=torch.float32)
    if aligned_target_logits is None:
        return zero, zero
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
    l1_dist_per_token = (draft_probs - target_probs).abs().sum(dim=-1)  # 逐位 TV 距离
    l1_loss_num = (l1_dist_per_token * loss_weight_mask).sum()  # 加权分子
    l1_loss_den = loss_weight_mask.sum()  # 权重总和（含指数衰减权重）作分母
    return l1_loss_num, l1_loss_den


def _collect_local_terms(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],  # 块内位置衰减 γ；None 不衰减
    l1_loss_alpha: float,  # L1 蒸馏权重；>0 才计算 L1 项
) -> tuple[dict[str, torch.Tensor], bool]:
    # 汇总本卡的各损失分子/分母与监控统计：
    # ①CE：展平后逐 token 交叉熵按 loss_weight_mask 加权；②L1 蒸馏（可选）；
    # ③置信度 BCE（可选，监督目标为 detach 的接受率 α）；另产出 no_grad 指标分子
    # （逐位接受率、τ、置信度误差/偏差）。返回 (loss_terms, has_confidence)。
    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    block_keep_mask = outputs.block_keep_mask
    _, _, block_size, vocab_size = draft_logits.shape  # 解包 [batch, num_blocks, block_size, vocab]
    device = draft_logits.device

    loss_weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=block_size,
        device=device,
        loss_decay_gamma=loss_decay_gamma,
    )  # 监督权重（可选指数衰减）
    flat_logits = draft_logits.reshape(-1, vocab_size)  # [B*num_blocks*block_size, vocab]
    flat_targets = target_ids.reshape(-1)  # 对应标签 token id
    flat_weights = loss_weight_mask.reshape(-1)  # 对应逐位权重
    loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # 逐 token CE（不做规约）
    ce_loss_num = (loss_per_token * flat_weights).sum()  # 加权分子
    ce_loss_den = flat_weights.sum()  # 权重和作分母（全局化待 all_reduce）
    aligned_target_logits = outputs.aligned_target_logits
    accept_rate_3d = _compute_accept_rate_3d(
        outputs=outputs,
        aligned_target_logits=aligned_target_logits,
    )  # 逐位接受率 α（可能为 None）
    zero = ce_loss_num.new_zeros(())
    assert l1_loss_alpha <= 0 or aligned_target_logits is not None, (
        "aligned_target_logits is required when l1_loss_alpha > 0."
    )  # 开 L1 蒸馏必须提供目标对齐 logits
    if l1_loss_alpha > 0:
        l1_loss_num, l1_loss_den = _compute_local_l1_term(
            outputs=outputs,
            aligned_target_logits=aligned_target_logits,
            loss_weight_mask=loss_weight_mask,
        )  # L1 蒸馏分子/分母
    else:
        l1_loss_num = zero
        l1_loss_den = zero

    with torch.no_grad():  # 以下全部是监控统计，不参与反向传播
        pos_total_counts = eval_mask.to(torch.float32).sum(dim=(0, 1))  # 逐位监督 token 数（accept_rate@pos 的分母）
        valid_pred_tokens = eval_mask.any(dim=-1)  # 块内是否存在监督位
        valid_blocks = block_keep_mask & valid_pred_tokens  # 有效块：非 dummy 且有监督
        valid_block_weights = valid_blocks.to(torch.float32)
        accept_block_count = valid_block_weights.sum()  # τ 指标的分母（有效块总数）
        tau_prob_sum, pos_accept_sums = _compute_local_probabilistic_stats(
            outputs=outputs,
            accept_rate_3d=accept_rate_3d,
            valid_block_weights=valid_block_weights,
        )  # 概率口径 τ 与逐位 α 求和

    has_confidence = outputs.confidence_pred is not None
    confidence_loss_num = zero
    confidence_loss_den = zero
    confidence_abs_error_num = zero
    confidence_bias_num = zero
    confidence_cumprod_bias_num = zero
    if has_confidence:
        assert accept_rate_3d is not None, (
            "aligned_target_logits is required when confidence head is enabled."
        )  # 置信度监督依赖 α，α 又依赖目标对齐 logits
        confidence_targets = accept_rate_3d.detach()  # 监督目标：接受率真值（detach，不回传到 draft 主干）
        confidence_errors = F.binary_cross_entropy_with_logits(
            outputs.confidence_pred.float(),
            confidence_targets,
            reduction="none",
        ) * loss_weight_mask  # 逐位 BCEWithLogits 再乘权重
        confidence_loss_num = confidence_errors.sum()
        confidence_loss_den = loss_weight_mask.sum()
        with torch.no_grad():
            confidence_probs = outputs.confidence_pred.float().sigmoid()  # logit → 概率
            confidence_error = confidence_probs - accept_rate_3d  # 逐位校准误差
            confidence_abs_error_num = (
                confidence_error.abs() * loss_weight_mask
            ).sum()  # 平均绝对误差分子（看校准精度）
            confidence_bias_num = (confidence_error * loss_weight_mask).sum()  # 有符号偏差（正=高估接受率）
            valid_mask = outputs.eval_mask.to(torch.float32)
            confidence_prefix_probs = (
                confidence_probs * valid_mask
            ).cumprod(dim=-1)  # 前缀接受概率 Π confidence——置信度调度器实际使用的量
            confidence_prefix_targets = (
                accept_rate_3d * valid_mask
            ).cumprod(dim=-1)  # 前缀真值 Π α
            confidence_cumprod_bias_num = (
                (confidence_prefix_probs - confidence_prefix_targets)
                * loss_weight_mask
            ).sum()  # 前缀积偏差：直接反映按置信度调度时的产出误差
    loss_terms = {  # 本卡分子/分母集合：分母待全局化，分子保留本地图
        "ce_loss_num": ce_loss_num,
        "ce_loss_den": ce_loss_den,
        "l1_loss_num": l1_loss_num,
        "l1_loss_den": l1_loss_den,
        "confidence_loss_num": confidence_loss_num,
        "confidence_loss_den": confidence_loss_den,
    }

    for pos_idx in range(block_size):  # 按块内位置分桶记录接受率（观察误差随深度复利的情况）
        add_metric(
            f"accept_rate@{pos_idx}",
            pos_accept_sums[pos_idx],  # 分子：该位置 α 之和
            den=pos_total_counts[pos_idx],  # 分母：该位置监督 token 数
            tag="train",
        )
    add_metric(
        "tau_probabilistic",
        tau_prob_sum,  # 分子：各有效块期望产出 token 数之和
        den=accept_block_count,  # 分母：有效块数 → 平均每块期望产出
        tag="train",
    )
    if has_confidence:
        add_metric(
            "confidence_abs_error",
            confidence_abs_error_num,
            den=confidence_loss_den,
            tag="train",
        )
        add_metric(
            "confidence_bias",
            confidence_bias_num,
            den=confidence_loss_den,
            tag="train",
        )
        add_metric(
            "confidence_cumprod_bias",
            confidence_cumprod_bias_num,
            den=confidence_loss_den,
            tag="train",
        )
    return loss_terms, has_confidence


def _build_loss(
    *,
    loss_terms: dict[str, torch.Tensor],  # 本卡分子/分母
    global_denominators: dict[str, torch.Tensor],  # all_reduce 后的全局分母
    ce_loss_alpha: float,  # CE 权重
    l1_loss_alpha: float,  # L1 蒸馏权重
    confidence_head_alpha: float,  # 置信度 BCE 权重
    has_confidence: bool,  # 本步是否真的有置信度输出
    world_size: int,  # 乘回 world_size：DDP 梯度平均后等价于全局均值损失
) -> torch.Tensor:
    # 组装反向传播用的标量损失：本卡分子 / 全局分母，三项加权后 × world_size。
    # 乘 world_size 的原因：DDP 对各卡梯度取平均，mean_i(num_i / den_global × W) 在 W=world_size 时
    # 等于 Σ_i num_i / den_global——恰为全局均值损失的梯度，各卡由此等价地优化同一全局目标。
    ce_loss = loss_terms["ce_loss_num"] / (global_denominators["ce_loss_den"] + 1e-6)  # +1e-6 防全批权重为 0
    l1_loss = ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:  # 全局 L1 分母为 0 说明本步全体无 L1 监督位
        l1_loss = loss_terms["l1_loss_num"] / (
            global_denominators["l1_loss_den"] + 1e-6
        )
    confidence_loss = ce_loss.new_zeros(())
    if has_confidence:
        confidence_loss = loss_terms["confidence_loss_num"] / (
            global_denominators["confidence_loss_den"] + 1e-6
        )
    return (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    ) * world_size  # 加权合成并放大，抵消 DDP 的梯度平均


def compute_dspark_loss(
    *,
    outputs: DSparkForwardOutput,  # draft 前向产物
    loss_decay_gamma: Optional[float],  # 块内位置衰减 γ
    ce_loss_alpha: float,  # CE 权重
    l1_loss_alpha: float,  # L1 蒸馏权重
    confidence_head_alpha: float,  # 置信度损失权重
):
    # 损失计算总入口：先算本卡分子/分母并刷监控指标，再 all_reduce 全局分母，
    # 返回用于 backward 的"局部分子 / 全局分母 × world_size"损失（分布式等价性见 _build_loss）。
    loss_terms, has_confidence = _collect_local_terms(
        outputs=outputs,
        loss_decay_gamma=loss_decay_gamma,
        l1_loss_alpha=float(l1_loss_alpha),
    )
    world_size = dist.get_world_size()
    global_denominators = _all_reduce_loss_denominators(
        loss_terms,
        world_size=world_size,
    )
    ce_loss_alpha = float(ce_loss_alpha)  # CLI 传入的字符串参数统一转 float
    l1_loss_alpha = float(l1_loss_alpha)
    confidence_head_alpha = float(confidence_head_alpha)

    local_ce_loss = loss_terms["ce_loss_num"] / (loss_terms["ce_loss_den"] + 1e-6)  # 纯本卡口径，仅用于日志对比
    local_l1_loss = local_ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:
        local_l1_loss = loss_terms["l1_loss_num"] / (
            loss_terms["l1_loss_den"] + 1e-6
        )
    local_confidence_loss = local_ce_loss.new_zeros(())
    if has_confidence:
        local_confidence_loss = loss_terms["confidence_loss_num"] / (
            loss_terms["confidence_loss_den"] + 1e-6
        )
    local_loss = (
        ce_loss_alpha * local_ce_loss
        + l1_loss_alpha * local_l1_loss
        + confidence_head_alpha * local_confidence_loss
    )  # 本卡加权损失：只作日志观察，不用于反向

    add_metric(
        "ce_loss",
        loss_terms["ce_loss_num"],  # 分子
        den=loss_terms["ce_loss_den"],  # 分母（flush 时全局相除）
        tag="train",
    )
    if global_denominators["l1_loss_den"].item() > 0:
        add_metric(
            "l1_loss",
            loss_terms["l1_loss_num"],
            den=loss_terms["l1_loss_den"],
            tag="train",
        )
    if has_confidence:
        add_metric(
            "confidence_loss",
            loss_terms["confidence_loss_num"],
            den=loss_terms["confidence_loss_den"],
            tag="train",
        )
    add_metric(
        "loss",
        local_loss,  # reduction="mean"：直接对本卡标量取均值，不再传分母
        reduction="mean",
        tag="train",
    )
    backward_loss = _build_loss(
        loss_terms=loss_terms,
        global_denominators=global_denominators,
        ce_loss_alpha=ce_loss_alpha,
        l1_loss_alpha=l1_loss_alpha,
        confidence_head_alpha=confidence_head_alpha,
        has_confidence=has_confidence,
        world_size=world_size,
    )  # 全局口径的反向损失
    return backward_loss


__all__ = [
    "compute_dspark_loss",
]
