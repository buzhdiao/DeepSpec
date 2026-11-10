# Eagle3 draft 的公共工具层：前向输出容器、目标层抽取校验、多层上下文特征拼接，
# 以及 Eagle3 TTT 训练所依赖的 FlexAttention/BlockMask 编译友好封装与专用注意力掩码。
from __future__ import annotations  # 延迟求值类型注解：允许标注引用尚未定义的类型

from dataclasses import dataclass  # 声明式数据容器：Eagle3ForwardOutput
from typing import Optional  # 可空类型注解

import torch  # PyTorch：张量操作、设备与编译接口
import torch._dynamo as dynamo  # TorchDynamo 配置：重编译缓存上限等
from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # FlexAttention：mask_mod 可编程稀疏注意力及其 BlockMask 构建器

from transformers.utils import is_torchdynamo_compiling  # 判断当前是否正处于 dynamo 追踪（外层 forward 被编译）


# Eagle3 draft 一次前向的输出容器：训练循环据此在 TTT 各步间传递隐状态并取 logits。
@dataclass
class Eagle3ForwardOutput:
    hidden_states: torch.Tensor  # draft 最后一层隐状态 [batch, seq, hidden]——TTT 下一步的特征输入
    draft_logits: torch.Tensor  # draft 词表头输出 [batch, seq, vocab]
    target_logits: Optional[torch.Tensor] = None  # 目标模型同位置 logits（蒸馏/评测对齐时才计算）


def validate_eagle3_target_layer_ids(layer_ids, num_target_layers: int):
    # 校验并规范化"从目标模型抽哪些层"的配置：Eagle3 v1 固定取 5 层隐状态拼接成
    # draft 输入特征，层号必须落在目标模型层数范围内且严格递增（保证特征有序拼接）。
    layer_ids = [int(layer_id) for layer_id in layer_ids]  # 容忍配置写成字符串/浮点，统一转 int
    assert len(layer_ids) == 5, (
        "Eagle3 v1 expects exactly 5 target layers, "
        f"got {len(layer_ids)}: {layer_ids}"
    )
    previous = None
    for layer_id in layer_ids:
        assert 0 <= layer_id < int(num_target_layers), (
            f"target_layer_id {layer_id} is out of range [0, {num_target_layers - 1}]"
        )
        assert previous is None or layer_id > previous, (
            "target_layer_ids must be strictly increasing."
        )
        previous = layer_id
    return layer_ids


def extract_eagle3_context_feature(hidden_states, layer_ids):
    # Eagle3 v1 only consumes target decoder layers. DSpark supports -1 for
    # embeddings, but that is intentionally out of scope here.
    # 提取 Eagle3 多层上下文特征：从目标模型各层输出元组中按层号取解码层输出，
    # 沿特征维拼接为 [batch, seq, len(layer_ids)*hidden]，交给 draft 的 fc 低秩投影压回 hidden。
    # hidden_states 元组下标 0 是 embedding 输出，故 decoder 层输出对应 layer_id+1。
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def configure_eagle3_flex_compile():
    # TTT 每多缓存一步，BlockMask 的形状（q_len×kv_len×lck 组合）就换一次，各触发一次重编译；
    # 把 dynamo 重编译缓存上限抬到 64，避免长训练中因编译缓存抖动反复回退 eager。
    if dynamo.config.recompile_limit < 64:
        dynamo.config.recompile_limit = 64


_COMPILED_FLEX_ATTENTION = None  # flex_attention 编译单例（懒初始化）
_COMPILED_CREATE_BLOCK_MASK = None  # create_block_mask 编译单例（懒初始化）


# 取编译版 flex_attention 的模块级单例；本函数自身禁用 dynamo 追踪，
# 使外层 torch.compile 不会把 torch.compile 调用本身再捕获进图。
# Adapted from SpecForge/specforge/modeling/draft/flex_attention.py:
# compile_friendly_flex_attention and compile_friendly_create_block_mask.
@torch.compiler.disable(recursive=False)
def get_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        configure_eagle3_flex_compile()
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)  # 首次调用编译一次，后续复用
    return _COMPILED_FLEX_ATTENTION


def compile_friendly_flex_attention(
    query: torch.Tensor,  # 查询 [batch, heads, q_len, head_dim]
    key: torch.Tensor,  # 键 [batch, kv_heads, kv_len, head_dim]
    value: torch.Tensor,  # 值 [batch, kv_heads, kv_len, head_dim]
    **kwargs,  # 透传 block_mask、enable_gqa、scale 等 flex_attention 参数
) -> torch.Tensor:
    # 编译友好的 flex_attention 入口：外层 forward 正被 dynamo 追踪时直接内联原始算子
    #（嵌套编译会图断裂/重复编译）；独立 eager 运行时改走已编译单例提速。
    flex_attention_func = (
        flex_attention if is_torchdynamo_compiling() else get_compiled_flex_attention()
    )
    return flex_attention_func(query, key, value, **kwargs)


# 取编译版 create_block_mask 单例（BlockMask 构建本身也是可编译计算图）；
# 与上者同样禁用自身追踪，避免在外层编译中嵌套 torch.compile。
@torch.compiler.disable(recursive=False)
def get_compiled_create_block_mask():
    # Adapted from SpecForge/specforge/modeling/draft/flex_attention.py:
    # WrappedCreateBlockMask.
    global _COMPILED_CREATE_BLOCK_MASK
    if _COMPILED_CREATE_BLOCK_MASK is None:
        configure_eagle3_flex_compile()
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
    return _COMPILED_CREATE_BLOCK_MASK


def compile_friendly_create_block_mask(
    mask_mod,  # 掩码谓词 (b, h, q_idx, kv_idx) -> bool
    B,  # batch 大小
    H,  # 头数（BlockMask 可按头广播，Eagle3 固定传 1）
    Q_LEN,  # 查询长度（当前 TTT 步块长）
    KV_LEN,  # 键值总长（含历史缓存）
    device,  # 构建 BlockMask 的目标设备
):
    # 与 compile_friendly_flex_attention 同策略的 BlockMask 构建入口。
    create_block_mask_func = (
        create_block_mask
        if is_torchdynamo_compiling()
        else get_compiled_create_block_mask()
    )
    return create_block_mask_func(mask_mod, B, H, Q_LEN, KV_LEN, device)


def create_eagle3_attention_mask(
    *,
    attention_mask: torch.Tensor,  # 2D padding 掩码 [batch, seq]，1 为有效 token
    q_len: int,  # 当前 TTT 步的块长（训练时=整条序列长）
    kv_len: int,  # 键值总长 = past_seen_tokens + q_len
    lck: int,  # 已缓存的 TTT 步数 = past_seen_tokens // q_len
    device: torch.device,  # BlockMask 目标设备
):
    # Adapted from SpecForge/specforge/modeling/draft/flex_attention.py:
    # generate_eagle3_mask.
    # 构建 Eagle3 TTT 训练专用 FlexAttention BlockMask，模拟推理时 draft 自回归起草的可见性。
    # KV 布局为 [历史 TTT 步缓存; 当前步]（DynamicCache 先旧后新）：
    #  - causal 项：query 因果可见最旧缓存步（chunk 0）有效前缀内的全部 key——基准上下文；
    #  - suffix 项：对后续每个缓存步（含当前步）只可见与 query 同一 chunk 内相对位置的
    #    对角 key——同一槽位在更深层 TTT 步的 draft 表征。
    # 两项合计恰好让 query（第 s 步槽位 i，对应原 token i+s+1）覆盖原 token 1..i+s+1
    # 各一条 draft KV，等价于推理时 draft cache 的因果前缀。
    seq_lengths = attention_mask.to(device=device).sum(dim=-1).to(torch.long)  # 每行有效 token 数 [batch]
    seq_lengths = (seq_lengths - int(lck)).clamp_min(0)  # 每缓存一步输入左移一位，有效 query 前缀随之收缩一格（上游注：causal 按对角线收缩）

    def eagle3_mask_mod(b, h, q_idx, kv_idx):
        del h  # 掩码与注意力头无关（H=1 构建后按头广播）
        seq_len = seq_lengths[b]  # 该行的有效 query 前缀长度
        in_valid_query = q_idx < seq_len  # 有效前缀之外的 query（移入的 padding 位）整体屏蔽
        causal_mask = (q_idx >= kv_idx) & (kv_idx < seq_len)  # 对最旧缓存步的有效前缀做因果可见
        suffix_mask = (
            (kv_idx >= q_len)  # 落在 chunk 0 之后的历史区（lck>=1 时也覆盖当前步自身 key）
            & ((kv_idx % q_len) < seq_len)  # 该 key 的块内相对位置仍在有效前缀内
            & (((kv_idx - q_idx) % q_len) == 0)  # 对角线：与 query 同一块内相对位置
        )
        return in_valid_query & (causal_mask | suffix_mask)

    eagle3_mask_mod.__name__ = f"eagle3_mask_Q_{q_len}_KV_{kv_len}_lck_{lck}"  # 名字带形状参数，便于 dynamo 按 TTT 形状区分缓存
    create_block_mask_func = (
        create_block_mask if int(q_len) <= 128 else compile_friendly_create_block_mask  # 小块直接构建（绕开编译缺陷 pytorch#160018），大块走编译版提速
    )
    return create_block_mask_func(
        eagle3_mask_mod,
        B=attention_mask.shape[0],  # batch 维
        H=1,  # 单头构建，按头广播
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
    )


def prepare_4d_causal_attention_mask(
    *,
    attention_mask: torch.Tensor,  # 2D padding 掩码 [batch, seq]
    dtype: torch.dtype,  # 加性掩码的浮点类型（无效处填其最小值）
    q_len: int,  # 查询长度
    kv_len: int,  # 键值总长
    past_seen_tokens: int,  # 已缓存 token 数（决定查询的绝对位置起点）
    device: torch.device,  # 张量目标设备
):
    # 非 flex 注意力路径的标准 4D 加性掩码 [batch, 1, q_len, kv_len]：因果 + padding。
    min_value = torch.finfo(dtype).min  # 该浮点型下确界，softmax 后权重归零
    query_positions = torch.arange(
        past_seen_tokens,
        past_seen_tokens + q_len,
        device=device,
    ).view(q_len, 1)  # 查询绝对位置 [q_len, 1]
    key_positions = torch.arange(kv_len, device=device).view(1, kv_len)  # 键绝对位置 [1, kv_len]
    causal = torch.where(
        key_positions <= query_positions,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )  # 因果项：允许看见自身与更早位置
    causal = causal.view(1, 1, q_len, kv_len)  # 升维以便按 batch/头广播

    expanded_mask = attention_mask[:, None, None, :kv_len].to(device=device).bool()  # padding 掩码 [batch, 1, 1, kv]
    padding = torch.where(
        expanded_mask,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )  # padding key 位置置最小值
    return causal + padding  # 两项相加得最终加性掩码


def eagle3_prepare_position_ids(
    *,
    input_ids: Optional[torch.Tensor] = None,  # 输入 token id [batch, seq]（与 embeds 二选一）
    input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入 [batch, seq, hidden]
    past_key_values_length: int = 0,  # 已缓存长度：位置从它起算
) -> torch.LongTensor:
    # 生成 [batch, seq] 连续位置 id（arange(past, past+seq) 按 batch 广播）；
    # TTT 的跨步 RoPE 偏移由模型侧 rope_cache_step_offset 另行处理。
    reference = input_ids if input_ids is not None else input_embeds  # 任取一个输入推断 batch/seq/设备
    assert reference is not None, "input_ids or input_embeds must be provided."
    batch_size, seq_len = reference.shape[:2]
    device = reference.device
    position_ids = torch.arange(
        int(past_key_values_length),
        int(past_key_values_length) + int(seq_len),
        dtype=torch.long,
        device=device,
    )
    return position_ids.unsqueeze(0).expand(batch_size, -1)  # [1, seq] -> [batch, seq]（各行共享同一位置序列）


__all__ = [
    "Eagle3ForwardOutput",
    "validate_eagle3_target_layer_ids",
    "extract_eagle3_context_feature",
    "configure_eagle3_flex_compile",
    "get_compiled_flex_attention",
    "compile_friendly_flex_attention",
    "get_compiled_create_block_mask",
    "compile_friendly_create_block_mask",
    "create_eagle3_attention_mask",
    "prepare_4d_causal_attention_mask",
    "eagle3_prepare_position_ids",
]
