# Qwen3 骨干的 DSpark draft 模型：跨注意力注入目标模型隐藏态，anchor 块内并行预测 + Markov 头半自回归修正。
from typing import Callable, Optional

import torch  # PyTorch：张量操作、CUDA 设备属性与同步
from torch import nn

from transformers.cache_utils import Cache  # KV cache 基类：推理期增量缓存 draft 自身的 K/V
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,  # 按实现名（sdpa/flash_attention_2/flex_attention…）分发注意力函数的注册表
    FlashAttentionKwargs,  # flash/sdpa 注意力的额外关键字参数类型（配合 Unpack 解包）
    GradientCheckpointingLayer,  # 按层做梯度检查点的基类：前向不存激活、反向重算，省显存换算力
    Qwen3MLP,  # Qwen3 前馈层（SwiGLU），draft 直接复用
    Qwen3PreTrainedModel,  # Qwen3 预训练模型基类：权重初始化/保存加载等基础设施
    Qwen3RMSNorm,  # RMSNorm（无均值中心化），Qwen3 各处归一化用
    Qwen3RotaryEmbedding,  # RoPE 旋转位置编码表
    eager_attention_forward,  # 朴素 eager 注意力实现（显式 softmax），作默认分发目标
    rotate_half,  # RoPE 的半向量旋转算子
)
from typing_extensions import Tuple, Unpack  # 旧版 Python 兼容的泛型别名与 kwargs 解包标注

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,  # 置信度头：线性预测逐位置接受率
    DSparkForwardOutput,  # 训练前向输出容器
    build_eval_mask,  # 生成连续前缀式逐位监督掩码
    create_dspark_attention_mask,  # 上下文前缀 + 块内自注意的块稀疏 BlockMask
    create_noise_embed,  # 构造 mask 占位 + anchor token 的 draft 输入嵌入
    create_position_ids,  # draft 区的绝对位置 id（anchor_pos + 块内偏移）
    log_sampler_stats,  # anchor 抽样健康度指标
    sample_anchor_positions,  # 无放回均匀采样 anchor 位置
)
from deepspec.modeling.dspark.markov_head import build_markov_head  # 按配置构建 vanilla/gated/rnn Markov 修正头
from deepspec.utils.sampling import sample_tokens  # 温度采样（temperature≈0 时 argmax）


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 本文件独立的 RoPE 应用：位置表按 [上下文 + draft] 全长生成，q 只取末尾 q_len 个位置
    # （查询仅对应 draft 区），k 覆盖全长（含目标上下文与 draft 区）。
    cos = cos.unsqueeze(unsqueeze_dim)  # 插入头维 [B, 1, S, D] 以便对多头广播
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])  # q 取末尾位置做旋转
    k_embed = (k * cos) + (rotate_half(k) * sin)  # k 用全位置旋转
    return q_embed, k_embed


class Qwen3DSparkAttention(nn.Module):
    # DSpark 跨注意力：query 来自 draft 隐藏态，K/V 来自"目标上下文隐藏态 ⊕ draft 自身隐藏态"拼接——
    # 每个 draft 位都能看见目标模型的上文表征（这是 DSpark 吃目标特征的核心通道），同时保留块内
    # 自注意以建模块内依赖。可见性由 create_dspark_attention_mask 的 BlockMask 控制（非因果）。
    def __init__(self, config, layer_idx: int):  # layer_idx：本层在 draft 中的层号，KV cache 按层分槽
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )  # 每头维度：新版配置显式给出，旧配置按 hidden_size 均分
        self.num_attention_heads = config.num_attention_heads  # 查询头数
        self.num_key_value_heads = config.num_key_value_heads  # KV 头数（GQA，通常少于查询头数）
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )  # 每个 KV 头服务的查询头组数
        self.scaling = self.head_dim**-0.5  # softmax 缩放因子 1/√d
        self.attention_dropout = config.attention_dropout  # 训练期注意力 dropout
        self.is_causal = False  # 块内并行预测，默认非因果（块稀疏掩码负责可见性）
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )  # 查询投影：draft 隐藏态 → num_heads * head_dim
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )  # 键投影：同时作用于目标上下文与 draft 隐藏态
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )  # 值投影：同上
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )  # 输出投影回 hidden_size
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Qwen3 特有：Q/K 逐头 RMSNorm 稳定注意力尺度
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )  # 仅滑窗层记录窗口大小；draft 配置全为 full_attention，故通常为 None

    def forward(
        self,
        hidden_states: torch.Tensor,  # draft 查询输入 [batch, q_len=num_blocks*block_size, hidden]
        target_hidden_states: torch.Tensor,  # 目标上下文特征 [batch, ctx_len=seq_len, hidden]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)：全长位置表
        attention_mask: Optional[torch.Tensor],  # BlockMask（flex 路径）或普通加性掩码
        past_key_values: Optional[Cache] = None,  # 推理期 KV cache（仅缓存 draft 自身 K/V）
        cache_position: Optional[torch.LongTensor] = None,  # cache 写入位置
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]  # 上下文长度 = 源序列长度
        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )  # [B, q_len, H, D]：拆查询头
        q = self.q_norm(q).transpose(1, 2)  # 逐头归一后转 [B, H, q_len, D]
        k_ctx = self.k_proj(target_hidden_states)  # 目标上下文的键
        k_noise = self.k_proj(hidden_states)  # draft 自身的键
        v_ctx = self.v_proj(target_hidden_states)  # 目标上下文的值
        v_noise = self.v_proj(hidden_states)  # draft 自身的值
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )  # KV = 上下文段 ⊕ draft 段，拆 KV 头
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)  # K 逐头 RMSNorm 后转 [B, Hkv, S, D]
        v = v.transpose(1, 2)  # V 不归一化，直接转 [B, Hkv, S, D]
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # q 取末尾位置、k 取全长应用 RoPE
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)  # 追加进 cache：返回含历史的全量 K/V
        if (
            self.config._attn_implementation == "flex_attention"
            and self.num_key_value_groups > 1
        ):
            kv_seq_len = k.shape[-2]
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)  # GQA 手工广播：每个 KV 头复制 groups 份（h ↔ kv h//groups）
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            k = k.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)  # 收回成与查询头同数的形状
            v = v.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
        attn_fn: Callable = eager_attention_forward  # 默认朴素实现
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # 按 sdpa/flash/flex 分发
        attn_is_causal = bool(kwargs.get("is_causal", False))
        # The SDPA path may consult module.is_causal when dispatching kernels,
        # so keep the per-call value mirrored on the module before invoking it.
        self.is_causal = attn_is_causal  # （原注释）把每次调用的因果标志同步到模块属性，供内核分发读取
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,  # 推理期无 dropout
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )  # 统一签名调用：首参为模块本体，供实现读取 num_key_value_groups 等属性
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )  # [B, q_len, H*D]：合并多头
        return self.o_proj(attn_output), attn_weights  # attn_weights 通常为 None（省显存）


class Qwen3DSparkDecoderLayer(GradientCheckpointingLayer):
    # 标准 pre-norm 解码层（attn → 残差 → norm → MLP → 残差），继承梯度检查点能力；
    # 与原版差异：注意力是跨注意（额外接收目标隐藏态），MLP/归一化直接复用 Qwen3 组件。
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DSparkAttention(config=config, layer_idx=layer_idx)  # 跨注意力子模块
        self.mlp = Qwen3MLP(config)  # Qwen3 SwiGLU 前馈层
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力前的 pre-norm
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )  # MLP 前的 pre-norm

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,  # 目标上下文特征（跨注意 K/V 源）
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states  # 残差分支保留原始输入
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]  # 跨注意 + 块内自注意
        hidden_states = residual + hidden_states  # 注意力残差回连
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states  # MLP 残差回连


class Qwen3DSparkModel(Qwen3PreTrainedModel):
    # DSpark draft 主体（Qwen3 版）：词嵌入/词头从目标拷贝，主干为少量 draft 层，
    # 通过 fc 融合目标多层特征作为跨注意上下文，配 Markov 修正头与置信度头。
    _no_split_modules = ["Qwen3DSparkDecoderLayer"]  # FSDP/设备映射时不跨设备切分的模块

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",  # 从目标取特征的层 id 列表
            "mask_token_id",  # draft 块内噪声占位 token
            "num_anchors",  # 每序列 anchor 块数
            "enable_confidence_head",  # 是否启用置信度头
            "markov_rank",  # Markov 头秩（0 为关闭）
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."  # DSpark 专有字段缺失即早失败
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), (
                "config.markov_head_type must be provided when markov_rank > 0."
            )  # 有秩必须有头类型
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when "
                "enable_confidence_head is true."
            )  # 开置信度头必须显式声明是否拼 Markov 特征
        self.target_layer_ids = config.target_layer_ids

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )  # 词嵌入：初始随机，训练前从目标模型拷贝（见 initialize_embeddings_and_head）
        self.layers = nn.ModuleList(
            [
                Qwen3DSparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )  # draft 解码层栈（层数 = num_draft_layers）
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 主干末端归一化
        self.rotary_emb = Qwen3RotaryEmbedding(config)  # RoPE：与目标共享位置编码方案
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )  # 特征融合：把目标多层拼接特征压缩回 hidden_size
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 融合特征的归一化
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 词头：从目标拷贝（tie=False 不共享）
        self.block_size = int(config.block_size)  # 每块 draft 位置数
        self.mask_token_id = config.mask_token_id  # 噪声占位 token id
        self.num_anchors = int(config.num_anchors)  # 每序列 anchor 块数

        # Markov head.
        self.markov_head = build_markov_head(config)  # Markov 修正头（rank=0 时为 None）

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None  # 拼 Markov 特征要求修正头已存在（共享其嵌入）

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank  # 拼接 Markov 低秩前驱嵌入时的输入维扩展
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)  # 接受率预测头
        self.post_init()  # HF 标准初始化（按 initializer_range 随机初始化并建立权重绑定）

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,  # 目标模型词嵌入
        lm_head: nn.Module,  # 目标模型词头（或与嵌入共享权重的对象）
        freeze: bool = True,  # 是否冻结拷贝后的参数
    ):
        # 从目标模型拷贝词嵌入与词头权重：draft 与目标共享同一词表示/输出空间是蒸馏对齐的前提；
        # 默认冻结（draft 只训练主干与各头，嵌入/词头保持目标分布不漂移）。
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape  # 词表/隐藏维必须一致
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)  # 切换词嵌入可训练性
        self.lm_head.requires_grad_(trainable)  # 切换词头可训练性

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)  # Qwen3 无 logit softcap，直接线性投影到词表

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,  # 当前步 draft 隐藏态 [.., hidden]
        prev_token_ids: Optional[torch.Tensor] = None,  # 前驱 token（with_markov 时必需）
    ) -> Optional[torch.Tensor]:
        # 推理期单步置信度预测：返回该位置接受率 logit（头未启用时返回 None）。
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )  # 前驱 token 的 Markov 低秩嵌入（对齐主干 dtype）
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)  # 特征拼接
            return self.confidence_head(features).float()  # 输出转 fp32 便于阈值比较
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,  # [batch, proposal_len, vocab] draft 主干输出的块 logits
        *,
        first_prev_token_ids: torch.Tensor,  # [batch] 块首前驱 token
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 推理期整块草稿采样：无修正头直接采样；有修正头则走半自回归逐步采样。
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )  # 空块占位
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits  # 无修正：logits 原样返回
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )  # 带 Markov 修正的逐步采样（返回采样 token 与修正后 logits）

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,  # [batch, vocab] 单步 logits
        *,
        prev_token_ids: torch.Tensor,  # [batch] 前驱 token id
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 推理期单步采样：先施加 Markov 偏置修正，再温度采样（供逐位调度式生成使用）。
        assert base_logits.ndim == 2, (
            "sample_draft_token_step expects base_logits shaped [batch, vocab], "
            f"got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )  # 修正后的单步 logits
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)  # [batch] 采样结果
        return sampled_token_ids, step_logits

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,  # [batch, ctx+draft] 全长位置 id（供 RoPE）
        attention_mask: Optional[torch.Tensor] = None,  # 块稀疏 BlockMask
        noise_embedding: Optional[torch.Tensor] = None,  # draft 输入嵌入
        target_hidden_states: Optional[torch.Tensor] = None,  # 目标多层拼接特征（待融合）
        past_key_values: Optional[Cache] = None,  # 推理期 KV cache
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # 共享前向主干：先把目标多层特征融合（fc 投影 + RMSNorm）为跨注意 K/V 源；
        # RoPE 表按全长位置一次生成、各层共用；随后逐层前向（支持按层梯度检查点）。
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))  # len(ids)*hidden → hidden，再归一
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # (cos, sin)：全长位置表一次计算
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )  # 每层：跨注意 + 块内自注意 → MLP 残差块
        return self.norm(hidden_states)  # 末端 RMSNorm 后输出

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len] 源序列（目标模型的真实输入）
        target_hidden_states: torch.Tensor,  # [batch, seq_len, len(ids)*hidden] 目标各层特征拼接
        loss_mask: torch.Tensor,  # [batch, seq_len] 逐 token 监督掩码
        target_last_hidden_states: Optional[torch.Tensor] = None,  # [batch, seq_len, hidden] 目标最后层隐藏态（对齐目标 logits 用）
    ) -> DSparkForwardOutput:
        # DSpark 训练前向全流程：
        # ①采样 anchor → ②构造噪声嵌入（块首放 anchor token）→ ③位置 id 与块稀疏掩码 →
        # ④backbone 前向（draft 查询同时看目标上下文与块内自身）→ ⑤构造标签与目标对齐 logits →
        # ⑥Markov 头教师强制修正 → ⑦置信度头预测 → 打包输出交 compute_dspark_loss。
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )  # ①无放回采样 anchor 位置与 dummy 掩码 [batch, num_anchors]
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )  # ②draft 输入嵌入 [batch, num_blocks*block_size, hidden]
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)  # 上下文段位置 0..seq_len-1
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)  # draft 段位置 anchor_pos + k
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)  # 全长位置表：供 RoPE 一次算出 (cos, sin)
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
        )  # ③块稀疏掩码：上下文前缀 + 块内自注意
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )  # ④主干前向 [batch, num_blocks*block_size, hidden]

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)  # [B, num_blocks, block_size, hidden]

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )  # 块内第 k 位预测 anchor+k+1（偏移 1..block_size）
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # 标签绝对位置 [B, num_blocks, block_size]
        safe_label_indices = label_indices.clamp(max=seq_len - 1)  # 越界标签钳到序列尾：gather 安全下标
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )  # dummy 块钳到 0：gather 结果随后被 eval_mask 屏蔽
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )  # [B, num_blocks, block_size] 按标签位置取出监督 token id
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)  # 标签 p 的目标分布应取目标在 p-1 处的隐藏态（下一 token 预测）
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ),
            )  # [B, num_blocks, block_size, hidden]：取与各草稿位对齐的目标隐藏态
            aligned_target_logits = self.compute_logits(aligned_target_hidden)  # ⑤目标侧对齐 logits（L1 蒸馏/置信度监督）
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )  # 连续前缀式逐位监督掩码
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )  # [B, num_blocks] 各块 anchor token（已置于噪声嵌入块首）
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )  # 教师强制前驱：第 k 位前驱 = 第 k-1 位真值标签（首位为 anchor token）
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )  # [B, num_blocks, block_size, vocab] 基础 logits
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )  # ⑥Markov 头教师强制修正（并行广播或逐位递归）

        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )  # 抽样健康度指标（no_grad）

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                    dtype=output_hidden_4d.dtype
                )  # 前驱 token 的 Markov 低秩嵌入作为置信度特征
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()  # ⑦[B, num_blocks, block_size] 接受率 logit
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )  # 打包输出，交由 compute_dspark_loss 计算损失


__all__ = [
    "Qwen3DSparkModel",
    "Qwen3DSparkAttention",
    "Qwen3DSparkDecoderLayer",
]
