# Qwen3 版 Eagle3 draft 模型：与目标模型同词嵌入/输出头，输入为目标多层隐状态的
# 低秩投影 + token embedding 的双流拼接；解码层沿用 Qwen3 结构（GQA、QK-Norm、
# 标准 pre-norm 残差），训练用 FlexAttention + Eagle3 TTT 专用掩码。
from __future__ import annotations  # 延迟求值类型注解

from typing import Callable, Optional

import torch  # PyTorch：张量与设备
from torch import nn  # 模块基类与层
from torch.nn.attention.flex_attention import flex_attention  # 可编程稀疏注意力算子

from transformers.cache_utils import Cache  # KV cache 协议（DynamicCache 等）
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,  # 按名称注册的注意力实现表（sdpa/flash_attention_2 等）
    FlashAttentionKwargs,  # flash-attn 路径的透传参数包
    GradientCheckpointingLayer,  # 按需触发梯度检查点的层基类
    Qwen3MLP,  # Qwen3 前馈层（SwiGLU）
    Qwen3PreTrainedModel,  # Qwen3 预训练模型基类（权重初始化/保存）
    Qwen3RMSNorm,  # Qwen3 RMSNorm
    Qwen3RotaryEmbedding,  # Qwen3 RoPE
    eager_attention_forward,  # eager 注意力实现（掩码路径参考实现）
    rotate_half,  # RoPE 的旋转半分量工具
)
from typing_extensions import Tuple, Unpack  # 兼容注解与 kwargs 解包标注

from deepspec.modeling.eagle3.common import (
    Eagle3ForwardOutput,  # draft 前向输出容器
    compile_friendly_flex_attention,  # 编译友好的 flex_attention 入口
    create_eagle3_attention_mask,  # Eagle3 TTT 专用 BlockMask
    eagle3_prepare_position_ids,  # 连续位置 id 生成
    prepare_4d_causal_attention_mask,  # 非 flex 路径的 4D 因果加性掩码
)
from deepspec.utils.sampling import sample_tokens  # 起草采样：贪心或温度多项采样


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 标准 RoPE：cos/sin 在头维前插入一维广播，对 q、k 做旋转位置编码。
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch, 1, seq, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 旋转 q
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 旋转 k
    return q_embed, k_embed


class Qwen3Eagle3Attention(nn.Module):
    # Qwen3 draft 注意力：q/k/v 从"embedding + fc 投影特征"的 2*hidden 拼接输入投影；
    # GQA + QK-Norm 与目标 Qwen3 一致，掩码由外部 BlockMask/加性掩码给定。
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)  # 层号：写 KV cache 时按层定位
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )  # 头维（缺省 hidden/头数）
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )  # GQA 组大小：每 KV 头对应的 query 头数
        self.scaling = self.head_dim**-0.5  # 1/sqrt(d) 缩放
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # Eagle3 掩码由外部给定，不走默认因果

        input_dim = int(config.hidden_size) * 2  # Eagle3 双流输入：token embedding 与 fc 投影特征拼接
        self.q_proj = nn.Linear(
            input_dim,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            input_dim,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            input_dim,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Qwen3 QK-Norm：逐头 RMSNorm
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = None  # draft 不用滑动窗口注意力

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, q_len, 2*hidden] 双流拼接输入
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # RoPE 的 (cos, sin)
        attention_mask: Optional[torch.Tensor],  # BlockMask（flex）或 4D 加性掩码（其他实现）
        past_key_values: Optional[Cache] = None,  # draft KV cache（TTT 跨步追加）
        cache_position: Optional[torch.LongTensor] = None,  # 当前块在 cache 中的绝对下标
        past_seen_tokens: int = 0,  # 已缓存 token 数
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)  # [batch, q, heads, dim]
        k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)  # [batch, q, kv_heads, dim]
        v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # QK-Norm 后转 [batch, heads, q, dim]
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # q、k 同时过 RoPE
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)  # 追加当前块，返回 [past; current]

        if self.config._attn_implementation == "flex_attention":
            # Direct flex_attention dispatch follows
            # SpecForge/specforge/modeling/draft/llama3_eagle.py.
            assert attention_mask is not None, (
                "Eagle3 flex_attention expects a BlockMask attention_mask."
            )
            flex_attention_func = (
                flex_attention
                if int(q_len) <= 128
                else compile_friendly_flex_attention  # 大块走编译版摊薄启动开销
            )
            attn_output = flex_attention_func(
                query=q,
                key=k.contiguous(),  # flex 要求连续内存
                value=v.contiguous(),
                block_mask=attention_mask,  # Eagle3 TTT 专用 BlockMask
                enable_gqa=True,  # KV 头由 kernel 内部按 GQA 广播（无需显式复制）
            )
            attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, q, heads, dim]
            attn_output = attn_output.reshape(bsz, q_len, -1)  # 合并头维
            return self.o_proj(attn_output), None

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # sdpa/flash_attention_2 等注册实现

        attn_is_causal = bool(
            kwargs.get(
                "is_causal",
                attention_mask is None and q_len > 1 and int(past_seen_tokens) == 0,  # 无掩码且整块 prefill 时才可用内置因果
            )
        )
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal  # 透传给注意力实现，避免其重复推断
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,  # 仅训练态启用 dropout
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)  # 合并头维
        return self.o_proj(attn_output), attn_weights


class Qwen3Eagle3DecoderLayer(GradientCheckpointingLayer):
    # Qwen3 draft 解码层：两流各自 RMSNorm 后拼接进注意力（双流尺度不同，分开归一），
    # 其余为 Qwen3 标准 pre-norm 残差结构。
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Eagle3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 特征流（fc 投影）的归一化
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # embedding 流的归一化（复用 Qwen3 命名）
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_embeds: torch.Tensor,  # [batch, q, hidden] 当前步 token embedding 流
        hidden_states: torch.Tensor,  # [batch, q, hidden] fc 特征流（残差基座）
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # 未使用（RoPE 已由 position_embeddings 提供）
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,  # 未使用：Eagle3 不返回注意力权重
        use_cache: Optional[bool] = False,  # 未使用：cache 写入由 attention 内部完成
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_seen_tokens: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        del position_ids, output_attentions, use_cache  # 显式丢弃未用参数
        residual = hidden_states  # 残差基座：fc 特征流
        hidden_states = self.hidden_norm(hidden_states)
        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)  # 双流拼接 [batch, q, 2*hidden]
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            past_seen_tokens=past_seen_tokens,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states  # 标准残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Eagle3Model(Qwen3PreTrainedModel):
    # Eagle3 draft 模型（Qwen3 版）：fc 低秩投影 + draft 解码层 + 与目标共享的
    # embedding/lm_head（由 initialize_embeddings_and_head 拷入并常冻结）。
    # Architecture adapted from
    # SpecForge/specforge/modeling/draft/llama3_eagle.py and
    # SpecForge/eval/model/eagle3.py.
    _no_split_modules = ["Qwen3Eagle3DecoderLayer"]  # accelerate 设备切分时层不跨卡

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",  # 抽层号（5 层）
            "ttt_length",  # TTT 步数
            "step_loss_decay",  # 步间损失衰减
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        self.target_layer_ids = [int(x) for x in config.target_layer_ids]
        self.ttt_length = int(config.ttt_length)
        self.step_loss_decay = float(config.step_loss_decay)

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )  # 词嵌入（权重从目标拷入）
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )  # Eagle3 低秩投影：5 层目标特征拼接压回 hidden，作为 draft 特征流
        self.layers = nn.ModuleList(
            [
                Qwen3Eagle3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 末层输出归一化
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 输出头权重从目标拷入
        self.post_init()  # HF 权重初始化入口

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,  # 目标模型的 embedding
        lm_head: nn.Module,  # 目标模型的输出头
        freeze: bool = True,
    ):
        # 把目标模型的 embedding 与 lm_head 权重拷入 draft（Eagle3 与目标共享词元嵌入
        # 和输出头）；freeze=True 时冻结二者，训练只更新 fc 与 draft 解码层。
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        # 切换 embedding 与 lm_head 的可训练性（Eagle3 常规做法是保持冻结）。
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # fc 低秩投影：[batch, q, len(target_layer_ids)*hidden] -> [batch, q, hidden]。
        assert hidden_states.size(-1) == len(self.target_layer_ids) * self.config.hidden_size
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # draft 出 logits：末层归一化 -> lm_head（头权重与目标一致）。
        return self.lm_head(self.norm(hidden_states))

    def draft_sample(self, logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        # 从 draft logits 起草下一 token：温度≈0 时贪心（argmax），否则按温度采样。
        return sample_tokens(logits, temperature=temperature)

    def _prepare_attention_mask(
        self,
        *,
        attention_mask: Optional[torch.Tensor],  # 2D padding 掩码（非 2D 时原样返回）
        hidden_states: torch.Tensor,  # 仅用于取设备/dtype
        q_len: int,  # 当前 TTT 步块长
        past_seen_tokens: int,  # 已缓存 token 数
    ):
        # 依注意力实现分派掩码：flex → Eagle3 TTT BlockMask；其他实现 → 标准 4D 因果加性掩码。
        if attention_mask is None or attention_mask.ndim != 2:
            return attention_mask
        kv_len = int(past_seen_tokens) + int(q_len)
        if self.config._attn_implementation == "flex_attention":
            assert int(past_seen_tokens) % int(q_len) == 0, (
                "Eagle3 flex_attention expects fixed-size TTT chunks: "
                f"past_seen_tokens={past_seen_tokens}, q_len={q_len}"
            )  # TTT 每步必须喂等长块，缓存按整块增长
            lck = int(past_seen_tokens) // int(q_len)  # 已缓存 TTT 步数
            return create_eagle3_attention_mask(
                attention_mask=attention_mask,
                q_len=q_len,
                kv_len=kv_len,
                lck=lck,
                device=hidden_states.device,
            )
        return prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            dtype=hidden_states.dtype,
            q_len=q_len,
            kv_len=kv_len,
            past_seen_tokens=int(past_seen_tokens),
            device=hidden_states.device,
        )

    def extend_draft_cache(
        self,
        hidden_states: torch.Tensor,  # 当前特征输入（上一步 draft 隐状态）
        input_ids: torch.LongTensor,  # 本步起草出的 token
        position_ids: torch.LongTensor,  # 本步位置
        past_key_values: Cache,  # draft KV cache（原地追加）
    ) -> torch.Tensor:
        # 推理辅助：把一个起草步喂进 draft 并写入其 KV cache，返回最后一个位置的
        # 隐状态 [batch, 1, hidden]——作为下一步起草的特征输入。
        assert input_ids.shape[1] > 0, "input_ids must contain at least one token."
        output = self(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return output[:, -1:, :]

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,  # fc 特征流：首步为 5 层目标拼接（未投影），后续为 draft 隐状态
        input_ids: Optional[torch.LongTensor] = None,  # 当前步输入 token（与 input_embeds 二选一）
        position_ids: Optional[torch.LongTensor] = None,  # 位置 id（缺省按 0 起连续生成）
        past_key_values: Optional[Cache] = None,  # draft KV cache
        use_cache: bool = False,  # 是否写 KV cache
        attention_mask: Optional[torch.Tensor] = None,  # 2D padding 掩码
        input_embeds: Optional[torch.Tensor] = None,  # 预算好的输入嵌入
        target_last_hidden_states: Optional[torch.Tensor] = None,  # 目标末层隐状态（算目标 logits）
        target_logits_only: bool = False,  # 只算目标 logits（蒸馏目标分布）
        return_logits: bool = False,  # 返回 Eagle3ForwardOutput 而非裸隐状态
        rope_cache_step_offset: bool = False,  # RoPE 位置加 lck（TTT 跨步对齐）
        **kwargs,
    ) -> torch.Tensor | Eagle3ForwardOutput:
        if target_logits_only:
            assert target_last_hidden_states is not None
            with torch.no_grad():  # 目标分布不回传梯度
                return self.lm_head(target_last_hidden_states)

        assert hidden_states is not None, "hidden_states must be provided."
        if hidden_states.size(-1) == len(self.target_layer_ids) * self.config.hidden_size:
            hidden_states = self.project_hidden_states(hidden_states)  # 未投影的多层拼接特征先过 fc

        if input_embeds is None:
            assert input_ids is not None, "Either input_ids or input_embeds must be provided."
            input_embeds = self.embed_tokens(input_ids)  # token embedding 流
        if position_ids is None:
            position_ids = eagle3_prepare_position_ids(
                input_ids=input_ids,
                input_embeds=input_embeds,
            )

        q_len = int(hidden_states.shape[1])
        past_seen_tokens = (
            int(past_key_values.get_seq_length())
            if past_key_values is not None
            else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + q_len,
            device=hidden_states.device,
        )  # 当前块在 KV cache 中的绝对下标
        prepared_attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            q_len=q_len,
            past_seen_tokens=past_seen_tokens,
        )
        rope_position_ids = position_ids
        if rope_cache_step_offset:
            assert int(past_seen_tokens) % int(q_len) == 0, (
                "SpecForge-style Eagle3 RoPE offset expects fixed-size TTT chunks: "
                f"past_seen_tokens={past_seen_tokens}, q_len={q_len}"
            )
            rope_position_ids = position_ids + int(past_seen_tokens) // int(q_len)  # 输入每步左移一位，+lck 拉回与原序列一致的 RoPE 绝对尺度
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        for layer in self.layers:
            hidden_states = layer(
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                attention_mask=prepared_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                past_seen_tokens=past_seen_tokens,
                **kwargs,
            )
        if return_logits:
            draft_logits = self.compute_logits(hidden_states)
            target_logits = None
            if target_last_hidden_states is not None:
                with torch.no_grad():  # 目标 logits 仅作蒸馏目标
                    target_logits = self.lm_head(target_last_hidden_states)
            return Eagle3ForwardOutput(
                hidden_states=hidden_states,  # TTT 下一步的特征输入
                draft_logits=draft_logits,
                target_logits=target_logits,
            )
        return hidden_states


__all__ = [
    "Qwen3Eagle3Model",
    "Qwen3Eagle3Attention",
    "Qwen3Eagle3DecoderLayer",
    "apply_rotary_pos_emb",
]
