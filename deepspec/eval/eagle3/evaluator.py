from __future__ import annotations  # 延迟注解求值：允许注解中使用 X | None 等新语法

from types import SimpleNamespace  # 属性式命名空间：draft 侧上下文状态

import torch  # PyTorch：张量操作
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # HF：自动加载模型/tokenizer 与 KV cache

from deepspec.eval.base_evaluator import (  # 基础评测框架
    BaseEvaluator,  # 评测器基类
    DraftProposal,  # 通用提案数据类
    VerificationResult,  # 验证结果数据类
    assert_no_final_target_layer,  # 校验 draft 抽取层不含 target 最后一层
    generate_decoding_sample,  # 投机解码生成主循环
    has_stop_token,  # 判断 token 是否命中 stop
)
from deepspec.modeling.eagle3 import extract_eagle3_context_feature  # 抽取 target 中间层隐状态特征（EAGLE3 约定）
from deepspec.modeling.eagle3.gemma4 import Gemma4Eagle3Model  # Gemma4 版 EAGLE3 draft 模型
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model  # Qwen3 版 EAGLE3 draft 模型
from deepspec.utils.sampling import logits_to_probs, sample_tokens  # 采样工具：概率转换与按温度采样


# Qwen3 目标模型的 EAGLE3 评测器：draft 自回归地逐个生成候选 token（复用 draft
# KV cache），target 一次性并行验证。EAGLE3 的 draft 以 target 中间层隐状态 +
# 上一 token 为输入（隐状态 i 与 token i+1 配对）。
class Qwen3Eagle3Evaluator(BaseEvaluator):
    EVAL_ATTN_IMPLEMENTATION = "sdpa"  # 评测统一用 sdpa 注意力实现
    draft_model_cls = Qwen3Eagle3Model  # draft 模型类；Gemma4 子类覆写为 Gemma4 版

    # 基类初始化后校验 draft 结构：单层 draft head 是 _update 无因果掩码扩展 cache 的前提
    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        # _update extends the draft cache with multiple committed tokens
        # without a causal mask. That is only safe when the draft head has a
        # single layer, so cached K/V are projected directly from per-token
        # inputs instead of from the previous layer's bidirectionally-mixed
        # attention output.
        draft_num_hidden_layers = int(self.draft_model.config.draft_num_hidden_layers)  # draft head 层数（须为 1，原因见上）
        assert draft_num_hidden_layers == 1, (
            f"{self.__class__.__name__} requires draft_num_hidden_layers == 1, "
            f"got {draft_num_hidden_layers}."
        )

    # 每轮提案上限 = draft 的 ttt_length（训练阶段设定的展开长度，即一次最多候选数）
    @property
    def max_proposal_tokens(self) -> int:
        return int(self.draft_model.ttt_length)

    # 加载 target 与 EAGLE3 draft 模型（bf16 + sdpa、eval 模式），统一层 id 类型并校验
    def build_models(self) -> tuple[object, Qwen3Eagle3Model, AutoTokenizer]:
        target_model = AutoModelForCausalLM.from_pretrained(  # target 模型：标准 HF CausalLM
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()

        draft_model = self.draft_model_cls.from_pretrained(  # draft 模型：EAGLE3 自定义结构
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        draft_model.target_layer_ids = [int(x) for x in draft_model.target_layer_ids]  # 层 id 统一成 int（checkpoint 里可能是其他类型）
        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)  # 抽取层不能是 target 最后一层

        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)  # tokenizer 与 target 对齐
        return target_model, draft_model, tokenizer

    # prefill 后初始化 draft 侧上下文：把 prompt 的 target 隐状态特征灌入 draft KV cache。
    # 错位约定：cache 槽位 i 存（隐状态 i，token i+1），RoPE 位置用 i——与训练时
    # "隐状态 i 配 token i+1"一致，保证训练/推理特征分布一致。
    def _init_context(
        self,
        *,
        initial_output,  # target prefill 输出（含 hidden_states）
        output_ids: torch.Tensor,  # 当前序列（prompt + 首个生成 token）
        position_ids: torch.Tensor,  # 全局位置 id
        num_input_tokens: int,  # prompt 长度
    ) -> SimpleNamespace:
        # Training pairs target hidden state i with token i + 1, while the
        # draft RoPE position stays at i.  Keep the same convention here when
        # pre-filling the draft cache from prompt hidden states.
        target_hidden = extract_eagle3_context_feature(  # prompt 各位置的 target 中间层特征（拼接多层）
            initial_output.hidden_states,
            self.draft_model.target_layer_ids,
        )
        shifted_prompt_ids = torch.cat(  # token 错位一移：位置 i 配 token i+1（末位补首个生成 token）
            [
                output_ids[:, 1:num_input_tokens],
                output_ids[:, num_input_tokens : num_input_tokens + 1],
            ],
            dim=1,
        )
        draft_cache = DynamicCache()  # draft KV cache 从零开始
        draft_hidden = self.draft_model.extend_draft_cache(  # 预填 cache，并得到最后位置的 draft 隐状态
            hidden_states=target_hidden,
            input_ids=shifted_prompt_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=draft_cache,
        )
        return SimpleNamespace(
            draft_cache=draft_cache,  # draft KV cache
            draft_hidden=draft_hidden,  # 最近位置的 draft 隐状态（下一轮自回归的起点）
            position_ids=position_ids,  # 保存全局位置 id 供后续提案/更新切片
            current_pos=num_input_tokens,  # 下一个待写入 token 对应的 cache 位置
            cache_len_before=0,  # 上一轮提案前的 cache 长度（首次为 0）
        )

    # 一轮提案（自回归 draft，逐 token 展开）：
    #   1. 记录提案前的 cache 长度（供 _update 回滚未提交候选的 KV）；
    #   2. 从当前 draft 隐状态出发循环 ttt_length 次：LM head 出 logits -> 按温度
    #      采样下一候选 -> 追加候选序列 -> 命中 stop 提前结束 -> 否则把新 token 写入
    #      draft cache 并前向一步，得到下一位的隐状态；
    #   3. 拼接全程 logits 与候选序列构成提案（候选首列为当前已接受 token）。
    # 与 dSpark 的整块并行提案不同，这是逐 token 自回归展开，逐轮复用 draft KV cache。
    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        # Adapted from SpecForge/eval/eval.py Eagle3 draft proposal loop.
        # cache_len_before is the draft-cache length *before* this proposal
        # extends it; _update needs it to crop back to that length.
        context.cache_len_before = context.draft_cache.get_seq_length()  # 记录提案前 cache 长度
        candidate_ids = [output_ids[:, start : start + 1]]  # 候选序列：当前 token 起头
        draft_logits_list = []  # 逐位置 draft logits（提案概率用）
        proposal_hidden = context.draft_hidden  # 自回归起点：最近位置的 draft 隐状态
        next_position = start  # 下一步写入 cache 用的 RoPE 位置（错位约定：token i+1 用位置 i）

        for _ in range(self.max_proposal_tokens):
            draft_logits = self.draft_model.compute_logits(proposal_hidden)  # draft LM head：隐状态 -> logits
            draft_logits_list.append(draft_logits)
            next_token = sample_tokens(  # 按温度采样下一候选 token
                draft_logits,
                temperature=float(self.args.temperature),
            )
            candidate_ids.append(next_token[:, -1:])  # 追加候选
            if has_stop_token(next_token, stop_token_ids):  # 命中 stop：提案到此为止
                break
            proposal_hidden = self.draft_model(  # 单步前向：新 token 写入 cache 并更新隐状态
                hidden_states=proposal_hidden,
                input_ids=next_token[:, -1:],
                position_ids=context.position_ids[
                    :,
                    next_position : next_position + 1,
                ],
                past_key_values=context.draft_cache,
                use_cache=True,
            )
            next_position += 1

        draft_logits = torch.cat(draft_logits_list, dim=1)  # 拼成 [B, k, V]
        return DraftProposal(
            draft_token_count=draft_logits.shape[1],  # 实际候选数（可能因 stop 提前结束）
            verify_input_ids=torch.cat(candidate_ids, dim=1),  # 验证输入：[当前 token] + 候选
            draft_probs=logits_to_probs(  # 逐位置 draft 概率 [B, k, V]，供拒绝采样
                draft_logits,
                float(self.args.temperature),
            ),
        )

    # 验证后更新 draft 侧状态（cache 回滚 + 重放已提交 token）：
    #   1. 把 draft cache 裁回提案前的长度——未提交候选的 KV 作废；
    #   2. 取本轮验证段的 target 中间层特征（截到提交长度），与已提交 token 按
    #      错位约定写入 cache（槽位 i 存 token i+1），前向得到新的 draft 隐状态；
    #   3. 推进 current_pos。这样被拒 token 的信息不会残留在 draft cache 里。
    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        # Adapted from SpecForge/eval/eval.py Eagle3 draft-cache crop/extend.
        assert verification.committed_tokens is not None
        committed_length = int(verification.committed_tokens.shape[1])  # 本轮提交 token 数（接受 draft + bonus）
        context.draft_cache.crop(int(context.cache_len_before))  # 回滚到提案前：未提交候选的 KV 作废
        committed_hidden = extract_eagle3_context_feature(  # 验证段各位置的 target 特征
            verification.target_output.hidden_states,
            self.draft_model.target_layer_ids,
        )[:, :committed_length, :]  # 截到提交长度（其后特征属于被拒位置）
        context.draft_hidden = self.draft_model.extend_draft_cache(  # 提交 token 写回 cache，得到新隐状态
            hidden_states=committed_hidden,
            input_ids=verification.committed_tokens,
            position_ids=context.position_ids[
                :,
                context.current_pos : context.current_pos + committed_length,
            ],
            past_key_values=context.draft_cache,
        )
        context.current_pos += committed_length  # 前进到提交末尾

    # 单样本生成：注入 init/propose/update 三个钩子（无 post_verify，EAGLE3 无置信度头）
    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,  # 编码后的 prompt [1, L]
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
        )


# Gemma4 目标模型的 EAGLE3 评测器：仅替换 draft 模型类，其余逻辑与 Qwen3 版完全一致
class Gemma4Eagle3Evaluator(Qwen3Eagle3Evaluator):
    draft_model_cls = Gemma4Eagle3Model


__all__ = ["Gemma4Eagle3Evaluator", "Qwen3Eagle3Evaluator"]  # 公开导出清单
