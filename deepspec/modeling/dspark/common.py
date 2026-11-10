# DSpark 共享工具：前向输出容器、目标层特征抽取、anchor 抽样、块稀疏注意力掩码、评测掩码与指标统计。
from dataclasses import dataclass  # DSparkForwardOutput 用 dataclass 承载一次训练前向的全部产物
from typing import Optional  # 可选张量字段的类型标注

import torch  # PyTorch：张量操作、CUDA 设备属性与同步
from torch import nn  # nn.Module：AcceptRatePredictor 等子模块基类
from torch.nn.attention.flex_attention import create_block_mask  # 把 mask_mod 函数编译成块稀疏 BlockMask（flex_attention 专用）

from deepspec.utils.metrics import add_metric  # 训练指标记录：ratio 型指标传分子/分母，flush 时全局 sum(num)/sum(den)


@dataclass
class DSparkForwardOutput:
    """Outputs for one DSpark training forward.

    Shape symbols:
        batch_size: number of samples in the batch
        seq_len: source sequence length
        num_anchors: sampled anchor blocks per sample
        block_size: number of draft positions per anchor
        vocab_size: vocabulary size

    The sampler keeps anchors whose first draft target is enabled by
    ``loss_mask``. Later slots are supervised only while they remain inside
    ``seq_len`` and form a contiguous enabled prefix. Dummy anchors can still
    appear when a sample has too few valid anchors; they are masked out by
    ``block_keep_mask`` and ``eval_mask``.
    """
    # 一次 DSpark 训练前向的输出容器：损失计算（compute_dspark_loss）所需的一切张量都从这里取。

    # [batch_size, num_anchors, block_size, vocab_size]
    draft_logits: torch.Tensor  # draft 对每个 anchor 块内 block_size 个位置的词表 logits（已含 Markov 头修正）
    # [batch_size, num_anchors, block_size]
    target_ids: torch.Tensor  # 监督标签：各 draft 位置对应 anchor 后续真实 token id
    # [batch_size, num_anchors, block_size]
    eval_mask: torch.Tensor  # 逐位监督掩码：块内为连续使能前缀，dummy 块与越界/无效位为 0
    # [batch_size, num_anchors]
    block_keep_mask: torch.Tensor  # 逐块保留掩码：False 表示该块是凑数的 dummy 块
    # [batch_size, num_anchors, block_size]
    confidence_pred: Optional[torch.Tensor] = None  # 置信度头输出 logit：预测该位置草稿被目标接受的概率
    # [batch_size, num_anchors, block_size, vocab_size]
    aligned_target_logits: Optional[torch.Tensor] = None  # 目标模型在对应预测位置上的 logits，供 L1 蒸馏与置信度监督


class AcceptRatePredictor(nn.Module):
    # 置信度头：单层线性回归，从 draft 隐藏态（可选拼接 Markov 隐变量）预测逐位置接受率 logit。
    # 训练目标是在线估计投机解码的逐位接受率 α = 1 - TV/2，推理期用于置信度调度（决定每块采纳多少草稿位）。
    def __init__(self, input_dim: int):  # 输入维：hidden_size，开启 with_markov 时再加 markov_rank
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)  # 输出标量 logit（无激活），配合 BCEWithLogits 训练

    def forward(self, features):  # features: [..., input_dim]，最后一维为特征维
        return self.proj(features).squeeze(-1)  # [.., 1] → [..]：返回逐位置 logit


def extract_context_feature(hidden_states, layer_ids):
    # 从目标模型输出元组按层抽取并拼接特征：hidden_states[0] 是 embedding 输出，因此 layer_id == -1
    # 映射下标 0，其余 layer_id 映射下标 layer_id + 1（第 layer_id 个 decoder 层）。
    # 返回 [batch, seq, len(layer_ids) * hidden_size]，作为 draft 跨注意的上下文特征（fc 融合前的原始拼接）。
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )  # 沿最后一维（特征维）按层顺序拼接


def validate_target_layer_ids(layer_ids, num_target_layers: int):
    # 校验目标层 id 列表：转 int、非空、每个 id 为 -1（embedding）或落在 [0, num_target_layers-1]，且严格递增。
    # 严格递增保证特征拼接顺序确定、权重可复现。
    layer_ids = [int(layer_id) for layer_id in layer_ids]  # 兼容 YAML/命令行传来的字符串或浮点
    assert layer_ids, "target_layer_ids must not be empty."
    start = 0
    end = int(num_target_layers) - 1  # 合法 decoder 层 id 上界
    previous = None
    for layer_id in layer_ids:
        assert layer_id == -1 or start <= layer_id <= end, (
            f"target_layer_id {layer_id} is out of range {{-1}} U [{start}, {end}] "
            f"for num_target_layers={num_target_layers}. "
            "-1 denotes the embedding output."
        )  # -1 是 embedding 输出的哨兵值，其余必须在层数范围内
        assert previous is None or layer_id > previous, (
            "target_layer_ids must be strictly increasing."
        )  # -1 < 任意 decoder 层 id，递增约束同时固定 embedding 在最前的拼接顺序
        previous = layer_id
    return layer_ids


def create_dspark_attention_mask(
    *,
    anchor_positions: torch.Tensor,  # [batch, num_blocks] 各 anchor 块在源序列中的起始位置
    block_keep_mask: torch.Tensor,  # [batch, num_blocks] False 为 dummy 块，整块屏蔽
    seq_len: int,  # 源序列长度：KV 划分为 [0, seq_len) 目标上下文段 + [seq_len, ...) draft 块区
    block_size: int,  # 每块 draft 位置数，用于行列下标与块 id 的换算
    device: torch.device,  # BlockMask 的落盘设备
):
    # 构造 DSpark 训练用的块稀疏注意力掩码（flex_attention 的 BlockMask）：
    # 查询只有 draft 位置（num_blocks * block_size 行），KV 分两段——前段目标上下文，后段 draft 自身。
    # 可见性规则：①上下文只允许看 anchor 之前的 token（kv_idx < anchor_pos），杜绝未来信息泄漏；
    # ②draft 区仅同块内互相可见（块内非因果自注意；anchor token 置于块首供全块共享）；
    # ③dummy 块整块屏蔽。块稀疏形式让 flex 内核直接跳过全屏蔽块以省算力。
    def dspark_mask_mod(b, h, q_idx, kv_idx):
        # flex_attention 回调：b 批、h 头（掩码与头无关）、q_idx/kv_idx 为展平后的行/列绝对下标。
        del h  # 所有多头共享同一掩码
        q_block_id = q_idx // block_size  # 查询行所属 anchor 块编号
        anchor_pos = anchor_positions[b, q_block_id]  # 该块 anchor 在源序列中的位置
        is_context = kv_idx < seq_len  # KV 落在目标上下文段
        mask_context = is_context & (kv_idx < anchor_pos)  # 上下文可见区间：anchor 之前（不含 anchor 本身）
        is_draft = kv_idx >= seq_len  # KV 落在 draft 块区
        kv_block_id = (kv_idx - seq_len) // block_size  # draft KV 所属块编号
        mask_draft = is_draft & (q_block_id == kv_block_id)  # draft 区仅同块可见（块内非因果）
        is_valid_block = block_keep_mask[b, q_block_id]  # dummy 块整体不可见
        return (mask_context | mask_draft) & is_valid_block  # 最终可见性：合法上下文 ∪ 同块 draft，且块有效

    bsz, num_blocks = anchor_positions.shape
    return create_block_mask(
        dspark_mask_mod,
        B=bsz,  # 逐批生成独立掩码（各批 anchor 不同）
        H=None,  # 头维共享
        Q_LEN=num_blocks * block_size,  # 查询行数：全部 draft 位置展平
        KV_LEN=seq_len + num_blocks * block_size,  # KV 列数：上下文 + draft 块区
        device=device,
    )


def build_anchor_candidate_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,  # [batch, seq_len] 逐 token 监督掩码（padding/无效位为 0）
) -> torch.Tensor:
    # 候选 anchor 判定：位置 p 可作 anchor 当且仅当 p 处 token 有效且首个草稿目标 p+1 也有效。
    # 前者保证 anchor token 是真实内容，后者保证块内第一个监督位有效（否则该块无训练信号，见 docstring）。
    num_candidates = max(seq_len - 1, 0)  # 候选区 [0, seq_len-2]：p+1 必须仍在序列内
    if num_candidates == 0:
        return loss_mask[:, :0].bool()  # 退化：序列过短，返回空候选掩码

    anchor_valid = loss_mask[:, :num_candidates] > 0.5  # anchor 位置自身有效
    first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5  # anchor+1（首个草稿目标）有效
    return anchor_valid & first_target_valid  # [batch, num_candidates] 布尔候选掩码


def sample_anchor_positions(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 从候选位无放回均匀抽取至多 num_anchors 个 anchor，并保证整批张量形状统一（批内对齐、可静态编译）：
    # 向量化实现——无效候选写随机值 2.0（>1，必排到队尾）与哨兵下标 seq_len+1，按随机键排序后有效候选
    # 以随机顺序排在最前；取前 num_anchors 个、按位置升序整理，不足处填 dummy（位置 0 + keep_mask=False）。
    # 返回 (anchor_positions [batch, num_anchors], block_keep_mask [batch, num_anchors])。
    valid = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
    )  # [batch, num_candidates] 候选掩码
    valid_counts = valid.sum(dim=1)  # 每条序列的有效候选数
    bsz = loss_mask.shape[0]
    num_candidates = valid.shape[1]
    max_n = int(num_anchors)
    if num_candidates == 0:
        anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)  # 全 0 位置占位
        keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)  # 全部标记为 dummy
        return anchors, keep_mask

    indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(
        bsz,
        -1,
    )  # [batch, num_candidates] 候选下标表
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    )  # 无效候选替换为越界哨兵 seq_len+1（之后被 keep_mask 掩掉）
    random_vals = torch.rand(bsz, num_candidates, device=device)  # U[0,1) 随机键
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))  # 无效候选键值 2.0，保证排到末尾
    _, sorted_idx = random_vals.sort(dim=1)  # 按随机键升序：有效候选被均匀打散并排在前段
    gathered = torch.gather(masked_indices, 1, sorted_idx)  # 排序后的候选下标（有效在前、随机顺序）
    if num_candidates < max_n:
        pad = torch.full(
            (bsz, max_n - num_candidates),
            seq_len + 1,
            dtype=gathered.dtype,
            device=device,
        )  # 候选总数不足时补哨兵位（dummy 块）
        gathered = torch.cat([gathered, pad], dim=1)
    anchors = gathered[:, :max_n].sort(dim=1).values  # 截断到 num_anchors 并按位置升序（真实 anchor 值小、自然排前）
    keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
        valid_counts.unsqueeze(1).clamp(max=max_n)
    )  # 前 min(valid_count, num_anchors) 个槽位是真实 anchor，其余为 dummy
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))  # dummy 槽位位置归零，配合 keep_mask 屏蔽
    return anchors, keep_mask


def build_eval_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    label_indices: torch.Tensor,  # [batch, num_blocks, block_size] 各草稿位监督的标签绝对位置
    safe_label_indices: torch.Tensor,  # 越界/dummy 已钳位后的安全 gather 下标
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    # 逐草稿位监督掩码：标签须在序列内、loss_mask 在标签处有效、块非 dummy；
    # 再沿块内维做前缀累积积（cumprod），使监督区形成"连续使能前缀"——一旦某位无效，
    # 其后所有位不再监督（半自回归语义下，后位失去可靠前驱，监督意义存疑且引入噪声）。
    target_valid = label_indices < seq_len  # 标签必须落在源序列内（块尾可能越界）
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
        2,
        safe_label_indices,
    )  # 按标签位置取 loss_mask（safe 下标保证 gather 不越界）
    eval_mask = target_valid & (target_loss_mask > 0.5)  # 位置合法且标签处受监督
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)  # dummy 块整体不监督
    return eval_mask.to(torch.int32).cumprod(dim=-1).bool()  # 前缀化：首个 False 之后全为 False


@torch.no_grad()
def log_sampler_stats(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    block_keep_mask: torch.Tensor,
    eval_mask: torch.Tensor,
    block_size: int,
    num_anchors: int,
) -> None:
    # 记录 anchor 抽样健康度指标（no_grad，纯监控，不建图不影响梯度）：
    # 平均每序列有效 anchor 数/占比、实采 anchor 数/占比、每块平均监督 token 数及其占 block_size 比例。
    valid_anchor_mask = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
    )  # 重新计算候选掩码（与抽样同一判定标准）
    valid_anchor_counts = valid_anchor_mask.sum(dim=1).to(torch.float32)  # 每序列有效 anchor 数
    valid_anchor_ratios = valid_anchor_counts / max(float(seq_len), 1.0)  # 有效 anchor 占序列长度比例
    sampled_anchor_counts = block_keep_mask.sum(dim=1).to(torch.float32)  # 实采到的非 dummy 块数
    sampled_anchor_ratios = sampled_anchor_counts / max(float(num_anchors), 1.0)  # 占配额比例（<1 说明候选不足）
    sample_count = loss_mask.new_tensor(float(loss_mask.shape[0]), dtype=torch.float32)  # 批大小，作 ratio 分母基准
    add_metric(
        "valid_anchors_abs",
        valid_anchor_counts.sum(),  # 分子：全批有效 anchor 总数
        den=sample_count,  # 分母：批大小 → 平均每序列有效 anchor 数
        tag="train",
    )
    add_metric(
        "valid_anchors_ratio",
        valid_anchor_ratios.sum(),  # 分子：占比之和
        den=sample_count,  # → 平均占比
        tag="train",
    )
    add_metric(
        "sampled_anchors_abs",
        sampled_anchor_counts.sum(),  # 分子：全批实采 anchor 总数
        den=sample_count,
        tag="train",
    )
    add_metric(
        "sampled_anchors_ratio",
        sampled_anchor_ratios.sum(),  # 分子：实采占配额比例之和
        den=sample_count,
        tag="train",
    )
    block_supervised_tokens = eval_mask.to(torch.float32).sum(dim=(1, 2)) / (
        sampled_anchor_counts.clamp_min(1.0)
    )  # 每序列平均每块的监督 token 数（clamp_min 防 dummy 全屏蔽时除零）
    add_metric(
        "block_supervised_tokens_abs",
        block_supervised_tokens.sum(),  # 分子：平均每块监督 token 数之和
        den=sample_count,
        tag="train",
    )
    add_metric(
        "block_supervised_tokens_ratio",
        (block_supervised_tokens / float(block_size)).sum(),  # 分子：监督 token 占 block_size 比例之和
        den=sample_count,
        tag="train",
    )


def create_position_ids(
    anchor_positions: torch.Tensor,  # [batch, num_blocks] anchor 绝对位置
    block_size: int,
) -> torch.Tensor:
    # 生成 draft 区的 RoPE 位置 id：块内第 k 位取 anchor_pos + k，使 draft 位置与源序列真实位置
    # 连续对齐（draft 与目标共享同一位置坐标系，RoPE 不漂移）。
    bsz, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)  # [1, 1, block_size] 块内偏移 0..block_size-1
    return (anchor_positions.unsqueeze(-1) + offsets).view(
        bsz,
        num_blocks * block_size,
    )  # 广播相加后展平为 [batch, num_blocks * block_size]


def create_noise_embed(
    embed_tokens: nn.Module,  # 与目标共享的词嵌入模块（权重已从目标拷贝）
    input_ids: torch.Tensor,  # [batch, seq_len] 源序列 token
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    mask_token_id: int,  # 噪声占位 token id
    block_size: int,
) -> torch.Tensor:
    # 构造 draft 输入嵌入：先铺满 mask_token（"待填噪声"占位），再把每块块首（偏移 0）替换为该块
    # anchor 的真实 token（dummy 块保持 mask token）。anchor token 进入块首使块内所有位置都能看到
    # 上文锚点（配合块稀疏掩码），其余位置保持 mask 供 draft 逐位预测 anchor 之后的 token。
    # 返回 [batch, num_blocks * block_size, hidden_size]。
    bsz = input_ids.shape[0]
    num_blocks = anchor_positions.shape[1]
    device = input_ids.device
    noise_ids = torch.full(
        (bsz, num_blocks * block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )  # 全 mask 占位 [batch, num_blocks * block_size]
    block_starts = torch.arange(num_blocks, device=device) * block_size  # 每块在 draft 区内的起始列
    block_starts = block_starts.unsqueeze(0).expand(bsz, -1)
    anchor_tokens = torch.gather(input_ids, 1, anchor_positions)  # [batch, num_blocks] 取各块 anchor 的真实 token
    flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(
        bsz,
        num_blocks,
    )  # 行索引表，供高级索引按 (batch, block) 坐标写入
    noise_ids[flat_batch_idx, block_starts] = torch.where(
        block_keep_mask,
        anchor_tokens,
        torch.tensor(mask_token_id, dtype=torch.long, device=device),
    )  # 非 dummy 块的块首写入真实 anchor token，dummy 块维持 mask
    return embed_tokens(noise_ids)  # 查词嵌入得到 draft 输入表示


__all__ = [
    "DSparkForwardOutput",
    "AcceptRatePredictor",
    "extract_context_feature",
    "validate_target_layer_ids",
    "create_dspark_attention_mask",
    "build_anchor_candidate_mask",
    "sample_anchor_positions",
    "build_eval_mask",
    "log_sampler_stats",
    "create_position_ids",
    "create_noise_embed",
]
