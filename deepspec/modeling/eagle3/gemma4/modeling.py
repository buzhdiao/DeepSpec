# Gemma4 版 Eagle3 draft 模型：与目标模型同词嵌入/输出头，输入为目标多层隐状态的
# 低秩投影 + token embedding 的双流拼接；解码层沿用 Gemma4 目标结构（QK-Norm、
# K=V 全局注意力变体、三明治归一化、logit softcap），训练用 FlexAttention + TTT 掩码。
from __future__ import annotations  # 延迟求值类型注解

from typing import Optional

import torch  # PyTorch：张量与设备
import torch.nn.functional as F  # scaled_dot_product_attention 等函数式接口
from torch import nn  # 模块基类与层
from torch.nn.attention.flex_attention import flex_attention  # 可编程稀疏注意力算子

from transformers.cache_utils import Cache  # KV cache 协议（DynamicCache 等）
from transformers.modeling_layers import GradientCheckpointingLayer  # 按需触发梯度检查点的层基类
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig  # Gemma4 文本配置类（draft config_class）
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4PreTrainedModel,  # Gemma4 预训练模型基类（权重初始化/保存）
    Gemma4RMSNorm,  # Gemma4 RMSNorm（可选带缩放）
    Gemma4TextMLP,  # Gemma4 MLP（含双宽变体）
    Gemma4TextRotaryEmbedding,  # Gemma4 RoPE（区分全局/局部层参数）
    Gemma4TextScaledWordEmbedding,  # Gemma4 嵌入：查表后乘 sqrt(hidden_size) 的缩放嵌入
    apply_rotary_pos_emb as apply_gemma4_rotary_pos_emb,  # Gemma4 版 RoPE：对单张量施加旋转（q、k 分别调用）
)

from deepspec.modeling.eagle3.common import (
    Eagle3ForwardOutput,  # draft 前向输出容器
    compile_friendly_flex_attention,  # 编译友好的 flex_attention 入口
    create_eagle3_attention_mask,  # Eagle3 TTT 专用 BlockMask
    eagle3_prepare_position_ids,  # 连续位置 id 生成
    prepare_4d_causal_attention_mask,  # 非 flex 路径的 4D 因果加性掩码
)
from deepspec.utils.sampling import sample_tokens  # 起草采样：贪心或温度多项采样


# FlexAttention 核参数：为 draft 小模型调小 tile 与 warp 数（前后向块均 32×32、4 warp），
# 在小头维/短块场景降低 kernel 开销。
GEMMA4_EAGLE3_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,  # 前向 query 块大小
    "BLOCK_N": 32,  # 前向 key/value 块大小
    "num_warps": 4,  # 前向 warp 数
    "bwd_BLOCK_M1": 32,  # 反向第一遍 query 块大小
    "bwd_BLOCK_N1": 32,  # 反向第一遍 kv 块大小
    "bwd_BLOCK_M2": 32,  # 反向第二遍 query 块大小
    "bwd_BLOCK_N2": 32,  # 反向第二遍 kv 块大小
}


class Gemma4Eagle3Attention(nn.Module):
    # Gemma4 draft 注意力：q/k/v 从"embedding + fc 投影特征"的 2*hidden 拼接输入投影；
    # attention_k_eq_v 时复用 k 作 v（Gemma4 全局层 K=V 变体）；缩放固定 1.0，与目标
    # dspark/gemma4 实现保持一致。
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)  # 层号：写 KV cache 时按层定位
        self.num_attention_heads = int(config.num_attention_heads)  # query 头数
        self.head_dim = int(config.global_head_dim)  # 头维取全局层头维
        self.use_alternative_attention = bool(config.attention_k_eq_v)  # K=V 变体开关
        if self.use_alternative_attention:
            self.num_key_value_heads = int(config.num_global_key_value_heads)  # K=V 变体用全局 KV 头数
        else:
            self.num_key_value_heads = int(config.num_key_value_heads)  # 常规 GQA 的 KV 头数
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads  # 每 KV 头对应的 query 头数（GQA 组大小）
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by the Gemma4 key/value head count."
        )
        self.scaling = 1.0  # 缩放系数固定 1.0（沿用 Gemma4 目标模型的注意力缩放约定）
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = False  # Eagle3 掩码由外部 BlockMask/加性掩码给定，不走默认因果

        input_dim = int(config.hidden_size) * 2  # Eagle3 双流输入：token embedding 与 fc 投影特征拼接
        self.q_proj = nn.Linear(
            input_dim,
            self.num_attention_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.k_proj = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.v_proj = None  # K=V 变体不建 v_proj
        if not self.use_alternative_attention:
            self.v_proj = nn.Linear(
                input_dim,
                self.num_key_value_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=bool(config.attention_bias),
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # QK-Norm：逐头 RMSNorm 稳定注意力
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            with_scale=False,  # Gemma4 的 v 归一化不带可学习增益
        )

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # GQA 展开：把 KV 头按组复制到 query 头数（本实现走显式复制而非 enable_gqa）。
        if self.num_key_value_groups == 1:
            return hidden_states
        return hidden_states.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, q_len, 2*hidden] 双流拼接输入
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # RoPE 的 (cos, sin)
        attention_mask: Optional[torch.Tensor],  # BlockMask（flex）或 4D 加性掩码（其他实现）
        past_key_values: Optional[Cache] = None,  # draft KV cache（TTT 跨步追加）
        cache_position: Optional[torch.LongTensor] = None,  # 当前块在 cache 中的绝对下标
        past_seen_tokens: int = 0,  # 已缓存 token 数
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        )  # [batch, q, heads, head_dim]
        k = self.k_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        )  # [batch, q, kv_heads, head_dim]
        if self.use_alternative_attention:
            v = k  # K=V 变体：值直接复用键
        else:
            v = self.v_proj(hidden_states).view(
                bsz,
                q_len,
                self.num_key_value_heads,
                self.head_dim,
            )

        q = self.q_norm(q).transpose(1, 2)  # QK-Norm 后转为 [batch, heads, q, dim]
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)

        cos, sin = position_embeddings
        q = apply_gemma4_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1)  # q、k 分别过 RoPE
        k = apply_gemma4_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)  # 追加当前块，返回 [past; current]

        k = self._repeat_kv(k)  # KV 头展开到 query 头数
        v = self._repeat_kv(v)
        if self.config._attn_implementation == "flex_attention":
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
                scale=self.scaling,
                kernel_options=GEMMA4_EAGLE3_FLEX_KERNEL_OPTIONS,
            )
        else:
            attn_is_causal = bool(
                kwargs.get(
                    "is_causal",
                    attention_mask is None
                    and q_len > 1
                    and int(past_seen_tokens) == 0,  # 无掩码且整块 prefill 时才可用内置因果
                )
            )
            self.is_causal = attn_is_causal
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,  # 仅训练态启用 dropout
                is_causal=attn_is_causal,
                scale=self.scaling,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, q, heads, dim]
        attn_output = attn_output.reshape(bsz, q_len, -1)  # 合并头维
        return self.o_proj(attn_output), None  # Eagle3 路径不返回注意力权重


class Gemma4Eagle3DecoderLayer(GradientCheckpointingLayer):
    # Gemma4 draft 解码层：双流输入（embedding + fc 特征）→ 注意力 → Gemma 式三明治
    # 归一化 MLP；残差基座是 fc 特征流，输出再乘逐层标量 layer_scalar。
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        assert not bool(config.enable_moe_block), (
            "Gemma4 Eagle3 prototype does not support Gemma4 MoE blocks yet."
        )  # draft 原型不支持 MoE 目标
        assert int(config.hidden_size_per_layer_input) == 0, (
            "Gemma4 Eagle3 prototype does not support per-layer input gates yet."
        )  # draft 原型不支持逐层输入门控
        self.self_attn = Gemma4Eagle3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.hidden_norm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )  # 特征流（fc 投影）的归一化
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )  # embedding 流的归一化（两流尺度不同，各自归一后再拼接）
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.register_buffer("layer_scalar", torch.ones(1))  # 逐层输出缩放（初始 1，与目标 Gemma4 一致）

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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_seen_tokens: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, output_attentions, use_cache  # 显式丢弃未用参数
        assert position_embeddings is not None, "position_embeddings must be provided."
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
        hidden_states = self.post_attention_layernorm(hidden_states)  # Gemma 式：注意力输出先归一再进残差
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)  # MLP 输出先归一再进残差
        hidden_states = residual + hidden_states
        return hidden_states * self.layer_scalar  # 逐层输出缩放


class Gemma4Eagle3Model(Gemma4PreTrainedModel):
    # Eagle3 draft 模型（Gemma4 版）：fc 低秩投影 + draft 解码层 + 与目标共享的
    # embedding/lm_head（由 initialize_embeddings_and_head 拷入并常冻结）。
    config_class = Gemma4TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4Eagle3DecoderLayer"]  # accelerate 设备切分时层不跨卡
    _supports_flex_attn = True  # 声明支持 flex_attention

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",  # 抽层号（5 层）
            "ttt_length",  # TTT 步数
            "step_loss_decay",  # 步间损失衰减
            "num_global_key_value_heads",  # Gemma4 全局 KV 头数
            "global_head_dim",  # Gemma4 全局头维
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        self.target_layer_ids = [int(x) for x in config.target_layer_ids]
        self.ttt_length = int(config.ttt_length)
        self.step_loss_decay = float(config.step_loss_decay)

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
            embed_scale=float(config.hidden_size) ** 0.5,  # Gemma 约定：查表结果乘 sqrt(hidden_size)
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )  # Eagle3 低秩投影：5 层目标特征拼接压回 hidden，作为 draft 特征流
        self.layers = nn.ModuleList(
            [
                Gemma4Eagle3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 末层输出归一化
        self.rotary_emb = Gemma4TextRotaryEmbedding(
            config,
            layer_type="full_attention",  # draft 全为全注意力层，取全局层 RoPE 参数
        )
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
        assert hidden_states.size(-1) == (
            len(self.target_layer_ids) * self.config.hidden_size
        )
        return self.fc(hidden_states)

    def _softcap_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Gemma4 logit softcap：tanh(logits/c)*c，把 logits 幅度压进 [-c, c]，
        # 与目标模型的输出分布约定保持一致。
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is None:
            return logits
        softcap = float(softcap)
        assert softcap > 0.0, (
            "config.final_logit_softcapping must be positive when provided."
        )
        return torch.tanh(logits / softcap) * softcap

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # draft 出 logits：末层归一化 -> lm_head -> softcap（头权重与目标一致）。
        return self._softcap_logits(self.lm_head(self.norm(hidden_states)))

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
                logits = self.lm_head(target_last_hidden_states)
                return self._softcap_logits(logits)

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
        position_embeddings = self.rotary_emb(
            hidden_states,
            rope_position_ids,
            layer_type="full_attention",
        )

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
                    target_logits = self._softcap_logits(
                        self.lm_head(target_last_hidden_states)
                    )
            return Eagle3ForwardOutput(
                hidden_states=hidden_states,  # TTT 下一步的特征输入
                draft_logits=draft_logits,
                target_logits=target_logits,
            )
        return hidden_states


__all__ = [
    "Gemma4Eagle3Model",
    "Gemma4Eagle3Attention",
    "Gemma4Eagle3DecoderLayer",
]
