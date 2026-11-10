from __future__ import annotations  # 延迟注解求值：允许注解中使用 X | None 等新语法

from dataclasses import dataclass  # dataclass 装饰器：声明提案结构体

import torch  # PyTorch：张量操作
from transformers import DynamicCache  # HF transformers：draft 侧 KV cache（支持按长度裁剪）

from deepspec.eval.base_evaluator import DraftProposal  # 通用提案基类
from deepspec.utils.sampling import logits_to_probs  # logits -> 概率分布（温度缩放 + softmax）
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel  # Gemma4 版 dSpark draft 模型
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # Qwen3 版 dSpark draft 模型


DSparkModel = Qwen3DSparkModel | Gemma4DSparkModel  # 两种 dSpark 实现的联合类型别名


# dSpark 提案：在通用提案之上附加置信度头输出，供评测期做逐位置校准统计
@dataclass
class DSparkDraftProposal(DraftProposal):
    confidence_logits: torch.Tensor | None = None  # 置信度头逐位置 logits [B, k]（sigmoid 后为"前缀到该位仍被接受"的预估）


# 跑一遍 dSpark draft backbone：输入 [当前 token + mask token 块] 与 target 中间层隐状态，
# 一次非自回归前向并行产出整块位置的隐状态（块内全可见，is_causal=False）。
# 返回的隐状态供采样与置信度预测；退出前把 draft KV cache 裁剪到 start——
# 未被验证采纳的块尾 KV 作废，下一轮提案重新计算。
def forward_dspark_draft_block(
    model: DSparkModel,  # dSpark draft 模型
    *,
    draft_input_ids: torch.Tensor,  # draft 输入 token [B, block_size]：首列为当前 token，其余为 mask token
    position_ids: torch.Tensor,  # 全局位置 id [1, L]
    past_key_values_draft: DynamicCache,  # draft KV cache（复用已验证前缀）
    target_hidden_states: torch.Tensor,  # target 中间层隐状态特征 [B, S, H]，draft 的条件输入
    start: int,  # 当前生成起始位置（下一个待验证 token 的全局下标）
    block_size: int,  # 一次前向产出的 token 块大小
) -> torch.Tensor:
    draft_position_ids = position_ids[  # 块内各位置的位置 id：从 cache 已有长度切片到 start + block_size
        :, past_key_values_draft.get_seq_length() : start + block_size
    ]
    block_hidden = model._forward_backbone(  # 一次前向并行算整块隐状态（多 token 预测，非逐 token 展开）
        target_hidden_states=target_hidden_states,
        noise_embedding=model.embed_tokens(draft_input_ids),  # 输入 token 嵌入（含 mask token 的占位嵌入）
        position_ids=draft_position_ids,
        attention_mask=None,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,  # 块内不做因果掩码：draft 一次看到整块上下文
    )
    past_key_values_draft.crop(start)  # 裁回 start：本轮提案未验证的 KV 不留在 cache 里
    return block_hidden  # 整块隐状态 [B, block_size, H]


# 构造空提案（draft_token_count=0）：验证输入只含当前 token，退化为 target 单 token 采样
# （用于置信度过低或置信度头不可用等场景，本轮不提交任何 draft token）
def _empty_dspark_proposal(draft_input_ids: torch.Tensor) -> DSparkDraftProposal:
    return DSparkDraftProposal(
        draft_token_count=0,
        verify_input_ids=draft_input_ids[:, :1],  # 只保留当前 token
        draft_probs=None,
        confidence_logits=None,
    )


# 预测逐位置置信度 logits：把块隐状态与已采样 token 喂给置信度头，
# 第 i 个输出估计"前缀到位置 i 仍会被 target 接受"。prev_token_ids 做错位一移：
# 位置 i 的条件 token 是位置 i-1 的采样结果（首列沿用当前 token），与训练时的
# (隐状态, 下一 token) 配对约定一致。
def _predict_confidence_logits(
    model: DSparkModel,
    *,
    proposal_hidden_states: torch.Tensor,  # 块隐状态 [B, block_size, H]
    draft_input_ids: torch.Tensor,  # draft 输入 token（首列为当前 token）
    sampled_tokens: torch.Tensor,  # 已采样的整块 draft token [B, block_size]
    block_size: int,  # 块大小
) -> torch.Tensor | None:
    prev_token_ids = torch.cat(  # 错位拼接：位置 i 的条件 token = 位置 i-1 的采样结果
        [draft_input_ids[:, :1], sampled_tokens[:, :-1]],
        dim=1,
    )
    confidence_pred = model.predict_confidence_step(  # 置信度头逐位置前向
        proposal_hidden_states,
        prev_token_ids=prev_token_ids,
    )
    if confidence_pred is None:  # 模型未实现置信度头
        return None
    return confidence_pred.float().reshape(  # 展平成 [B, block_size, ?] 后取每位置第 0 维（单值 logit）
        confidence_pred.shape[0],
        block_size,
        -1,
    )[:, :, 0]


# 计算置信前缀长度：对逐位置置信 logits 取 sigmoid 得接受概率，
# 返回首个低于阈值的位置下标（即连续置信位置的个数）；全部置信则返回整块 block_size。
# threshold <= 0 表示不截断（信任整块提案）。
def _confident_prefix_length(
    confidence_logits: torch.Tensor,
    *,
    block_size: int,
    threshold: float,
) -> int:
    if threshold <= 0.0:  # 阈值为 0：不做置信截断
        return int(block_size)
    below_threshold = confidence_logits.sigmoid() < threshold  # 逐位置标记低置信位置
    if not bool(below_threshold[0].any().item()):  # 全部位置置信：整块提交
        return int(block_size)
    return int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())  # 第一个低置信位置即前缀长度


# 构造 dSpark 提案（评测的核心提案路径），流程：
#   1. 块隐状态过 LM head 得 base logits，逐位置采样整块 draft token；
#   2. 有置信度头时预测逐位置置信 logits，按阈值截出置信前缀长度 k（无置信度头则整块）；
#   3. k=0 或置信度头异常时退化为空提案（只验证当前 token，本轮零投机）；
#   4. 拼装验证输入 [当前 token + 前 k 个采样 token] 与对应 draft 概率分布。
# 返回的 confidence_logits 同步截到 k，与提交前缀对齐，供 post_verify 做校准统计。
def build_dspark_proposal(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,  # draft 输入 token（首列当前 token，其余 mask）
    block_hidden: torch.Tensor,  # draft 块前向输出的隐状态
    block_size: int,  # 块大小
    temperature: float,  # draft 采样温度
    confidence_threshold: float,  # 置信度截断阈值（0 表示不截断）
) -> DSparkDraftProposal:
    assert draft_input_ids.size(0) == 1, "build_dspark_proposal requires batch_size=1"  # 评测仅支持单序列
    proposal_hidden_states = block_hidden[:, :block_size, :]  # 取块内位置的隐状态
    base_draft_logits = model.compute_logits(proposal_hidden_states)  # LM head：隐状态 -> 词表 logits
    sampled_tokens, draft_logits = model.sample_draft_tokens(  # 逐位置采样整块 token（首位置条件于当前 token）
        base_draft_logits,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        hidden_states=proposal_hidden_states,
    )

    proposal_draft_tokens = int(block_size)  # 提案长度默认整块
    confidence_logits = None
    if model.confidence_head is not None:  # 模型带置信度头：先预测再按阈值截断
        confidence_logits = _predict_confidence_logits(
            model,
            proposal_hidden_states=proposal_hidden_states,
            draft_input_ids=draft_input_ids,
            sampled_tokens=sampled_tokens,
            block_size=block_size,
        )
        if confidence_logits is None:  # 置信度头未返回结果：退化为空提案
            return _empty_dspark_proposal(draft_input_ids)
        proposal_draft_tokens = _confident_prefix_length(  # 截出置信前缀长度 k
            confidence_logits,
            block_size=block_size,
            threshold=float(confidence_threshold),
        )

    if proposal_draft_tokens == 0:  # 置信前缀为空：本轮不提交 draft token
        return _empty_dspark_proposal(draft_input_ids)

    verify_input_ids = torch.cat(  # 验证输入 = [当前 token] + 前 k 个采样 token，形状 [B, 1+k]
        [draft_input_ids[:, :1], sampled_tokens[:, :proposal_draft_tokens]],
        dim=1,
    )
    draft_probs = logits_to_probs(  # 前 k 个位置的 draft 概率分布 [B, k, V]，供拒绝采样
        draft_logits[:, :proposal_draft_tokens, :],
        temperature,
    )
    return DSparkDraftProposal(
        draft_token_count=proposal_draft_tokens,
        verify_input_ids=verify_input_ids,
        draft_probs=draft_probs,
        confidence_logits=(  # 置信 logits 截到 k，与提交前缀逐位置对齐
            confidence_logits[:, :proposal_draft_tokens]
            if confidence_logits is not None
            else None
        ),
    )
