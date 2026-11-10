from __future__ import annotations  # 延迟注解求值：允许注解中使用 X | None 等新语法

import json  # 标准库：解析 JSONL 数据集行
import os  # 标准库：从模型路径取 basename 作为表格显示名
import random  # 标准库：按 seed 随机下采样数据集
from dataclasses import dataclass  # dataclass 装饰器：声明提案/验证结果结构体
from pathlib import Path  # 路径对象：拼接数据集文件与 TensorBoard 目录
from types import SimpleNamespace  # 属性式命名空间：包装生成结果与算法上下文
from typing import Any, Callable  # 类型注解：Any 与回调函数签名

import torch  # PyTorch：张量操作、随机数与设备管理
import torch.distributed as dist  # torch 分布式：多卡 all_reduce 汇总指标
from transformers import DynamicCache  # HF transformers：KV cache 容器（支持按长度 crop）

from deepspec.data.parser import encode_chat_messages  # 按 chat template 把对话消息编码为 input_ids
from deepspec.utils.sampling import (  # 采样工具集（概率转换、采样、拒绝采样残差修正）
    gather_token_probs,  # 按 token id 从概率分布中取对应概率值
    logits_to_probs,  # logits 经温度缩放 + softmax 得到概率分布
    sample_from_probs,  # 按概率分布采样 token
    sample_residual,  # 从归一化 max(0, p_target − p_draft) 重采样：拒绝位置的修正 token
)
from deepspec.utils import (  # 通用工具
    init_dist,  # 初始化分布式进程组，返回 device/global_rank/world_size
    seed_all,  # 统一设置 python/numpy/torch 随机种子，保证可复现
)


DEFAULT_DATASET_ROOT = "./eval_datasets"  # 数据集默认根目录（JSONL 文件所在路径）


# 读取 JSONL 数据集并做格式校验：每行必须含非空字符串列表字段 `turns`。
# 只保留首轮对话（turns[:1]）——评测固定按单轮生成；返回原始行字典列表。
def load_and_process_dataset(
    data_name: str,  # 数据集名（不含扩展名），对应 <root>/<data_name>.jsonl
    dataset_root: str = DEFAULT_DATASET_ROOT,  # 数据集根目录
):
    dataset_path = Path(dataset_root) / f"{data_name}.jsonl"  # 拼出数据集文件路径
    assert dataset_path.exists()  # 文件不存在直接断言失败

    rows = []  # 收集全部有效行
    with dataset_path.open("r", encoding="utf-8") as handle:  # 逐行读 JSONL
        for line_number, line in enumerate(handle, start=1):  # 记录行号便于报错定位
            line = line.strip()  # 去除首尾空白
            if not line:  # 跳过空行
                continue
            row = json.loads(line)  # 解析该行 JSON
            turns = row.get("turns")  # 取对话轮次字段
            assert (
                isinstance(turns, list)
                and len(turns) > 0
                and all(isinstance(turn, str) for turn in turns)
            ), (
                f"{dataset_path}:{line_number} must contain a non-empty string list field `turns`."
            )
            row["turns"] = turns[:1]  # 截断为单轮：评测只用第一条 user 消息
            rows.append(row)  # 加入结果
    return rows  # 返回全部数据行


# 截断生成序列：若输出中出现 stop token，则保留到第一个 stop token（含）为止。
# 只在生成段（num_input_tokens 之后）内查找，prompt 部分不受影响。
def trim_output_ids(
    output_ids: torch.Tensor,  # 生成序列（含 prompt），形状 [1, L]
    num_input_tokens: int,  # prompt 长度
    stop_token_ids: list[int] | None,  # stop token id 列表；None 表示无需截断
) -> torch.Tensor:
    if stop_token_ids is None:  # 未配置 stop token：原样返回
        return output_ids
    stop_token_tensor = torch.tensor(stop_token_ids, device=output_ids.device)  # stop id 搬到同设备
    stop_token_indices = torch.isin(  # 生成段内命中 stop token 的相对下标（升序）
        output_ids[0][num_input_tokens:],
        stop_token_tensor,
    ).nonzero(as_tuple=True)[0]
    if stop_token_indices.numel() == 0:  # 未命中：原样返回
        return output_ids
    return output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]  # 保留到首个 stop token（含）


# 判断 token 序列中是否出现任一 stop token（stop_token_ids 为 None 时恒为 False）
def has_stop_token(token_ids: torch.Tensor, stop_token_ids: list[int] | None) -> bool:
    if stop_token_ids is None:
        return False
    stop_token_tensor = torch.tensor(stop_token_ids, device=token_ids.device)  # stop id 搬到同设备
    return bool(torch.isin(token_ids, stop_token_tensor).any().item())  # 任一命中即 True


# 解析 stop token id 列表：优先取 target 模型 generation_config 的 eos_token_id，
# 回退到 tokenizer 的 eos；支持 int 或列表，列表去重后返回，全无则返回 None。
def resolve_stop_token_ids(target_model, tokenizer) -> list[int] | None:
    generation_config = getattr(target_model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)  # 先看模型 generation_config
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id  # 回退到 tokenizer
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]

    stop_token_ids = []  # 列表形式（多 EOS 场景）：逐个去重收集
    for token_id in eos_token_id:
        token_id = int(token_id)
        if token_id not in stop_token_ids:
            stop_token_ids.append(token_id)
    return stop_token_ids


# 校验 draft 抽取的 target 层不含最后一层 decoder。原因：transformers 的
# output_hidden_states 在最后一层位置存的是终归一化后的隐状态，而 target cache
# 生成路径存的是原始 decoder 层输出，二者不一致会导致 draft 的输入特征错位；
# 必须改用更早的层并重建 target cache 与 draft 权重。
def assert_no_final_target_layer(target_model, target_layer_ids) -> None:
    target_config = target_model.config
    if hasattr(target_config, "text_config"):  # 多模态模型：真实文本配置嵌套在 text_config
        target_config = target_config.text_config
    last_layer_id = int(target_config.num_hidden_layers) - 1  # 最后一层 decoder 的索引
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    assert last_layer_id not in target_layer_ids, (
        "target_layer_ids must not include the final target decoder layer "
        f"{last_layer_id}. Transformers output_hidden_states stores the final "
        "normalized hidden state at that position, while target cache generation "
        "stores raw decoder-layer outputs. Use an earlier layer and regenerate "
        "the target cache and draft checkpoint."
    )


# 用 prettytable 渲染评测结果汇总表：每行一个数据集，列 = 基础指标 + 逐位置接受率。
# 逐位置列数取所有数据集提案长度的最大值，短提案的缺失列用 "-" 补齐。
def build_results_table(
    *,
    rows: list[dict[str, object]],  # 各数据集指标行（build_metrics_row 的输出）
    model_name_or_path: str,  # target 模型路径（显示 basename）
    draft_name_or_path: str,  # draft 模型路径（显示 basename）
    header: bool = True,  # 是否输出表头（单数据集打印时关闭）
) -> str:
    from prettytable import PrettyTable  # 延迟导入：表格渲染库

    table = PrettyTable()
    max_positions = max(  # 逐位置接受率列的最大数量
        (len(metrics["accept_rates_by_position"]) for metrics in rows),
        default=0,
    )
    field_names = [  # 基础列：数据集、两个模型名、提案长度、accept length、验证位置利用率
        "dataset",
        "target_model",
        "draft_model",
        "#propose",
        "accept_len",
        "verify_rate",
    ]
    field_names.extend(f"accept_rate@{pos_idx}" for pos_idx in range(max_positions))  # 追加逐位置接受率列
    table.field_names = field_names
    table.header = header

    normalized_target_model_name = model_name_or_path.rstrip("/")  # 去掉路径尾斜杠
    target_model_name = (  # 取 basename 作显示名；根路径等情况回退为全路径
        os.path.basename(normalized_target_model_name) or normalized_target_model_name
    )
    normalized_draft_model_name = draft_name_or_path.rstrip("/")
    draft_model_name = (
        os.path.basename(normalized_draft_model_name) or normalized_draft_model_name
    )
    for metrics in rows:
        row = [  # 基础列：提案长度显示为 "n+1"（n 个 draft token + 1 个 bonus token）
            metrics["dataset"],
            target_model_name,
            draft_model_name,
            f"{metrics['draft_tokens_per_proposal']:.2f}+1",
            f"{metrics['acceptance_length']:.2f}",
            f"{metrics['verify_rate']:.4f}",
        ]
        row.extend(  # 逐位置接受率；None（该位置从未被提案）显示为 "-"
            f"{accept_rate:.4f}" if accept_rate is not None else "-"
            for accept_rate in metrics["accept_rates_by_position"]
        )
        row.extend(["-"] * (max_positions - len(metrics["accept_rates_by_position"])))  # 不足最大列数补 "-"
        table.add_row(row)
    return table.get_string()  # 返回渲染后的表格文本


# draft 提案数据类：一次验证请求携带的全部信息
@dataclass
class DraftProposal:
    draft_token_count: int  # 本提案的 draft token 数（不含前缀里的当前已接受 token）
    verify_input_ids: torch.Tensor  # 验证输入 [B, 1+k]，首列为当前已接受 token，后 k 列为 draft token
    draft_probs: torch.Tensor | None  # draft 各位置概率分布 [B, k, V]；None 则无法做拒绝采样（纯 target 采样）


# target 一次并行验证的结果数据类：接受统计 + 本轮提交 token
@dataclass
class VerificationResult:
    target_output: Any  # target 前向输出（含 logits/hidden_states，供 draft 侧提取上下文特征）
    target_probs: torch.Tensor  # target 各验证位置的概率分布 [B, verify_length, V]
    accept_prefix_mask: torch.Tensor | None  # 逐位置接受前缀掩码（cumprod），[B, k]：一旦某位被拒其后全 0
    accepted_draft_tokens: int  # 实际接受的 draft token 数（draft 命中 stop token 时提前截断）
    next_token: torch.Tensor  # bonus token [B, 1]：残差重采样或 target 直接采样得到
    effective_proposal_length: int  # 实际生效的提案长度（stop token 截断后可能小于 draft_token_count）
    terminated_by_stop_token: bool = False  # 是否因接受的 draft token 命中 stop token 而终止
    committed_tokens: torch.Tensor | None = None  # 本轮提交 token：接受的 draft token + bonus token [B, accepted+1]


# 用 target 模型对整块 draft token 做一次性并行验证（投机解码的核心步骤）。
# 验证序列 = [当前已接受 token] + draft token，共 draft_token_count + 1 个位置：
#   - 前 draft_token_count 个位置逐位置拒绝采样：以 min(1, p_target/p_draft) 的概率
#     接受该 draft token；接受前缀用 cumprod 保证"一旦拒绝其后全部作废"；
#   - 最后 1 个位置产生 bonus token：若某 draft 位置被拒，则从归一化残差分布
#     max(0, p_target − p_draft) 重采样（保证输出分布与纯 target 采样一致，无偏）；
#     全部接受则按 target 分布直接采样。
# 返回 VerificationResult：接受数、接受前缀掩码、bonus token 与提交序列；
# 副作用：target KV cache 写入整个验证段的 KV（由调用方 crop 回有效长度）。
def verify_draft_tokens(
    *,
    target_model,  # target 模型（HF CausalLM 风格前向接口）
    proposal: DraftProposal,  # 待验证提案
    position_ids: torch.Tensor,  # 全局位置 id [1, L]
    start: int,  # 验证段起始位置（当前 token 的全局下标）
    past_key_values_target: DynamicCache,  # target KV cache（原地写入验证段 KV）
    temperature: float,  # 采样温度（同时作用于 target/draft 概率转换）
    max_proposal_tokens: int,  # 提案长度上限
    current_token_ids: torch.Tensor | None = None,  # 当前已接受 token，用于校验提案前缀
    stop_token_ids: list[int] | None = None,  # stop token 列表
) -> VerificationResult:
    """Verify draft tokens with the target model and rejection sampling."""
    if proposal.draft_token_count > max_proposal_tokens:  # 提案长度不得超过上限
        raise ValueError(
            "DraftProposal.draft_token_count must not exceed "
            f"max_proposal_tokens={max_proposal_tokens}, "
            f"got {proposal.draft_token_count}."
        )
    if current_token_ids is not None and not torch.equal(
        proposal.verify_input_ids[:, :1],
        current_token_ids,
    ):
        raise ValueError(
            "DraftProposal.verify_input_ids must start with the current "
            "accepted token."
        )

    draft_token_count = int(proposal.draft_token_count)  # draft token 数 k
    verify_length = draft_token_count + 1  # 验证长度 = k 个 draft 位置 + 1 个 bonus 位置
    verify_position_ids = position_ids[:, start : start + verify_length]  # 验证段的 RoPE 位置 id
    target_output = target_model(  # target 一次前向并行验证整块 token（KV 写入 cache）
        input_ids=proposal.verify_input_ids,
        position_ids=verify_position_ids,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,  # 中间层隐状态供 draft 侧提取上下文特征
    )
    if target_output.logits.ndim != 3:  # 校验 logits 形状必须是 [B, S, V]
        raise ValueError(
            "target model must return rank-3 logits [B, S, V], "
            f"got ndim={target_output.logits.ndim}."
        )
    target_probs = logits_to_probs(target_output.logits, float(temperature))  # 温度缩放 + softmax 得 target 分布
    if (
        draft_token_count > 0
        and proposal.draft_probs is not None
        and proposal.draft_probs.size(-1) != target_probs.size(-1)
    ):
        raise ValueError(
            "DraftProposal.draft_probs vocab size must match target logits, "
            f"got {proposal.draft_probs.size(-1)} and {target_probs.size(-1)}."
        )

    accept_prefix_mask = None
    if draft_token_count > 0:  # 有 draft token 才做拒绝采样
        assert proposal.draft_probs is not None
        proposed_tokens = proposal.verify_input_ids[:, 1:]  # 各位置的 draft token [B, k]
        selected_target_probs = gather_token_probs(  # target 给这些 token 的概率 [B, k]
            target_probs[:, :-1, :],  # 前 k 个位置的分布（最后一列对应 bonus 位置，不参与）
            proposed_tokens,
        )
        selected_draft_probs = gather_token_probs(  # draft 给这些 token 的概率 [B, k]
            proposal.draft_probs,
            proposed_tokens,
        ).clamp_min(1e-8)  # 防止除零
        accept_prob = torch.clamp(  # 拒绝采样接受概率 min(1, p_target/p_draft)
            selected_target_probs / selected_draft_probs,
            max=1.0,
        )
        accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)  # 逐位置独立掷骰
        accept_prefix_mask = accept_mask.cumprod(dim=1)  # 前缀掩码：一旦某位置被拒，其后全部为 0
        accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())  # 最长接受前缀长度（bsz=1）
    else:
        accepted_draft_tokens = 0  # 空提案：只产生 bonus token

    effective_proposal_length = draft_token_count  # 生效提案长度初始为全长
    terminated_by_stop_token = False
    if stop_token_ids and accepted_draft_tokens > 0:  # 在已接受的 draft token 中查 stop token
        accepted_slice = proposal.verify_input_ids[0, 1 : accepted_draft_tokens + 1]  # 已接受的 draft token
        stop_tensor = torch.tensor(
            stop_token_ids,
            device=accepted_slice.device,
            dtype=accepted_slice.dtype,
        )
        eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]  # 命中位置（升序）
        if eos_hits.numel() > 0:  # 命中：stop token 之后的接受 token 一并作废
            eos_pos = int(eos_hits[0].item())
            accepted_draft_tokens = eos_pos + 1  # 保留到 stop token（含）
            effective_proposal_length = eos_pos + 1  # 生效提案长度同步截断
            terminated_by_stop_token = True

    if 0 < draft_token_count and accepted_draft_tokens < draft_token_count:  # 存在被拒位置
        assert proposal.draft_probs is not None
        next_token = sample_residual(  # bonus token 从残差分布重采样：归一化 max(0, p_target − p_draft)
            target_probs[:, accepted_draft_tokens, :],  # 被拒位置的 target 分布
            proposal.draft_probs[:, accepted_draft_tokens, :],  # 同位置的 draft 分布
        )
    else:  # 全部接受（或空提案）：bonus token 直接按 target 分布采样
        next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)  # 最后一个位置的分布 [B, V]

    committed_tokens = torch.cat(  # 本轮实际提交：接受的 draft token + bonus token [B, accepted+1]
        [
            proposal.verify_input_ids[:, 1 : accepted_draft_tokens + 1],
            next_token.unsqueeze(1),
        ],
        dim=1,
    )

    return VerificationResult(
        target_output=target_output,
        target_probs=target_probs,
        accept_prefix_mask=accept_prefix_mask,
        accepted_draft_tokens=accepted_draft_tokens,
        next_token=next_token,
        effective_proposal_length=effective_proposal_length,
        terminated_by_stop_token=terminated_by_stop_token,
        committed_tokens=committed_tokens,
    )


# 投机解码生成主循环（算法无关的通用框架），流程：
#   1. prefill：target 模型吃下整个 prompt，缓存 KV，按温度采样首个生成 token；
#   2. 循环（start < max_length）：draft 侧 propose 产出候选提案（含当前 token 前缀），
#      verify_draft_tokens 一次性并行验证，提交"已接受 draft token + 1 个 bonus token"；
#   3. 每轮用 crop 把 target KV cache 裁到实际提交长度，被拒 token 的 KV 不留残留。
# 四个钩子由具体算法（EAGLE3/dSpark）注入：
#   - init_context：prefill 后构建 draft 侧状态（draft KV cache、target 隐状态特征等）；
#   - propose：产出下一个 DraftProposal；
#   - update：验证通过后推进 draft 侧状态（循环继续时调用）；
#   - post_verify：可选诊断钩子，每轮验证后调用（用于置信度头校准统计）。
# 仅支持 batch_size=1；返回结果附带逐轮统计：
#   acceptance_lengths（每轮 accept length = 接受 draft 数 + bonus）、
#   proposal_lengths（每轮生效提案长度）、accepted_draft_lengths（每轮接受 draft 数）。
@torch.inference_mode()  # 全程推理模式：禁用梯度追踪，省显存提速度
def generate_decoding_sample(
    *,
    target_model,  # target 模型
    input_ids: torch.Tensor,  # 编码后的 prompt [1, L]
    max_new_tokens: int,  # 最大生成 token 数
    max_proposal_tokens: int,  # 每轮提案长度上限
    temperature: float,  # 采样温度
    stop_token_ids: list[int] | None,  # stop token 列表
    init_context: Callable[..., Any],  # 钩子：构建算法私有上下文
    propose: Callable[..., DraftProposal],  # 钩子：产出下一提案
    update: Callable[[Any, VerificationResult], None],  # 钩子：验证后推进上下文
    post_verify: Callable[[DraftProposal, VerificationResult], None] | None = None,  # 可选诊断钩子
) -> SimpleNamespace:
    """Speculative-decoding loop.

    `init_context(initial_output, output_ids, position_ids, num_input_tokens)`
    builds the algorithm-specific state once after prefill. `propose(context,
    output_ids, position_ids, start, stop_token_ids)` returns the next
    DraftProposal. `update(context, verification)` advances the state when the
    loop continues. `post_verify` is an optional diagnostic hook called after
    every verification (used for confidence calibration).
    """
    assert max_proposal_tokens >= 1  # 至少允许提案 1 个 token
    assert input_ids.size(0) == 1, "only bsz=1 is supported"  # 评测只支持单序列

    device = input_ids.device
    num_input_tokens = input_ids.shape[1]  # prompt 长度
    max_length = num_input_tokens + int(max_new_tokens)  # 序列总长度上限

    output_ids = torch.empty(  # 预分配输出缓冲：多留提案空间，避免循环内扩容
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)  # 全局位置 id [1, L_max]
    past_key_values_target = DynamicCache()  # target KV cache，跨轮复用

    output = target_model(  # prefill：整个 prompt 一次前向，只保留最后 1 个位置的 logits
        input_ids=input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,  # 隐状态供 draft 初始化上下文特征
        logits_to_keep=1,
    )

    output_ids[:, :num_input_tokens] = input_ids  # 写入 prompt
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_from_probs(  # 按温度采样首个生成 token
        logits_to_probs(output.logits, float(temperature))
    )

    start = input_ids.shape[1]  # 下一个待生成 token 的全局位置
    acceptance_lengths: list[int] = []  # 每轮 accept length（接受的 draft 数 + 1 个 bonus）
    proposal_lengths: list[int] = []  # 每轮生效提案长度
    accepted_draft_lengths: list[int] = []  # 每轮接受的 draft token 数（不含 bonus）

    initial_token = output_ids[:, num_input_tokens : num_input_tokens + 1]  # 首个生成 token
    if has_stop_token(initial_token, stop_token_ids):  # 首 token 即 stop：无验证轮次，直接结束
        output_ids = output_ids[:, : num_input_tokens + 1]
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)  # 保留 stop token
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=output_ids.shape[1] - num_input_tokens,  # 生成 token 数
            acceptance_lengths=acceptance_lengths,  # 空
            proposal_lengths=proposal_lengths,  # 空
            accepted_draft_lengths=accepted_draft_lengths,  # 空
            verify_count=0,  # 未发生任何验证
        )

    context = init_context(  # 构建算法私有状态（draft KV cache、target 特征等）
        initial_output=output,
        output_ids=output_ids,
        position_ids=position_ids,
        num_input_tokens=num_input_tokens,
    )

    while start < max_length:  # 主循环：propose -> verify -> commit
        proposal = propose(  # draft 侧产出候选（验证输入首列为当前 token）
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
            stop_token_ids=stop_token_ids,
        )
        verification = verify_draft_tokens(  # target 并行验证整块提案
            target_model=target_model,
            proposal=proposal,
            position_ids=position_ids,
            start=start,
            past_key_values_target=past_key_values_target,
            temperature=temperature,
            max_proposal_tokens=max_proposal_tokens,
            current_token_ids=output_ids[:, start : start + 1],  # 当前 token，校验提案前缀一致性
            stop_token_ids=stop_token_ids,
        )
        if post_verify is not None:  # 诊断钩子：如置信度头逐位置校准统计
            post_verify(proposal, verification)

        proposal_lengths.append(int(verification.effective_proposal_length))  # 记录提案长度
        accepted_draft_tokens = int(verification.accepted_draft_tokens)  # 本轮接受的 draft token 数
        accepted_draft_lengths.append(accepted_draft_tokens)

        output_ids[:, start : start + accepted_draft_tokens + 1] = (
            proposal.verify_input_ids[:, : accepted_draft_tokens + 1]  # 提交：当前 token + 接受的 draft token
        )

        if verification.terminated_by_stop_token:  # draft 命中 stop token：收尾
            acceptance_lengths.append(accepted_draft_tokens)  # 无 bonus token，accept length 到 stop 为止
            start += accepted_draft_tokens
            past_key_values_target.crop(start)  # 裁掉被拒 token 的 KV
            break

        output_ids[:, start + accepted_draft_tokens + 1] = verification.next_token  # 写入 bonus token
        new_token_ids = output_ids[:, start + 1 : start + accepted_draft_tokens + 2]  # 本轮新提交的 token 段
        acceptance_lengths.append(accepted_draft_tokens + 1)  # accept length = 接受 draft 数 + 1（bonus）
        start += accepted_draft_tokens + 1  # 前进到 bonus token 之后
        past_key_values_target.crop(start)  # 裁掉被拒 token 的 KV，cache 与 start 对齐
        update(context, verification)  # 推进 draft 侧状态

        if has_stop_token(new_token_ids, stop_token_ids):  # 新提交 token 含 stop：结束生成
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]  # 截到实际生成长度（不超过上限）
    output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)  # 按 stop token 收尾截断
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_ids.shape[1] - num_input_tokens,  # 生成 token 数
        acceptance_lengths=acceptance_lengths,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_draft_lengths,
        verify_count=len(proposal_lengths),  # 验证轮数
    )


# 评测器基类：串联 数据加载 -> 投机解码生成 -> 分布式指标汇总 -> 表格/TensorBoard 上报。
# 子类只需实现 max_proposal_tokens / build_models / generate_one_sample 三个接口。
class BaseEvaluator:

    # 初始化分布式环境并构建 target/draft 模型与 tokenizer
    def __init__(self, local_rank: int, args):  # local_rank：本进程 GPU 序号；args：运行配置
        self.args = args  # 保存配置（tasks、seed、temperature、tensorboard_dir 等）
        self.device, self.global_rank, self.world_size = init_dist(local_rank)  # 初始化进程组
        self.tasks = args.tasks  # 待评测任务列表 [(数据集名, 最大样本数或 None)]

        self.target_model, self.draft_model, self.tokenizer = self.build_models()  # 由子类构建模型
        self.metrics_rows: list[dict[str, object]] = []  # 各数据集指标行，供最终汇总表格使用

    # 每轮提案的最大 draft token 数（dSpark 为 block_size，EAGLE3 为 ttt_length）
    @property
    def max_proposal_tokens(self) -> int:
        raise NotImplementedError  # 子类必须实现

    # 构建 target 模型、draft 模型与 tokenizer；子类实现
    def build_models(self):
        raise NotImplementedError  # 子类必须实现

    # 对单条样本执行一次投机解码生成；子类实现（内部调用 generate_decoding_sample）
    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,  # 编码后的 prompt [1, L]
        stop_token_ids: list[int] | None,  # stop token 列表
    ) -> SimpleNamespace:
        raise NotImplementedError  # 子类必须实现

    # 由分布式汇总的原始计数构建单数据集指标行（rank 0 调用）：
    #   acceptance_length = 总接受 token / 验证轮数（投机解码核心指标 accept length，含 bonus token）；
    #   draft_tokens_per_proposal = 总提案长度 / 验证轮数（表格里 "n+1" 的 n）；
    #   verify_rate = 总接受 token / 总验证位置数（每轮验证 k+1 个位置），衡量 target 前向利用率；
    #   accept_rates_by_position = 逐位置接受率：位置 i 的接受次数 / 提案长度超过 i 的轮数。
    def build_metrics_row(
        self,
        *,
        dataset_name: str,  # 数据集名
        metric_summary: dict[str, int | list[int]],  # all_reduce 后的全局汇总计数
    ) -> dict[str, object]:
        proposal_count = int(metric_summary["proposal_count"])  # 验证总轮数
        if proposal_count == 0:  # 无任何验证轮次：全部指标置零/占位
            acceptance_length = 0.0
            draft_tokens_per_proposal = 0.0
            verify_rate = 0.0
            accept_rates_by_position = [None] * self.max_proposal_tokens  # None 占位
        else:
            acceptance_length = (  # 平均每轮接受 token 数（含 bonus），即 accept length
                int(metric_summary["acceptance_length_sum"]) / proposal_count
            )
            draft_tokens_per_proposal = (  # 平均每轮提案长度
                int(metric_summary["proposal_length_sum"]) / proposal_count
            )
            verify_rate = int(metric_summary["acceptance_length_sum"]) / (  # 提交 token / 总验证位置（提案长 + 每轮 bonus 位）
                int(metric_summary["proposal_length_sum"]) + proposal_count
            )
            proposals_at_pos = metric_summary["proposals_at_pos"]  # 位置 i 出现在提案中的轮数
            accepted_at_pos = metric_summary["accepted_at_pos"]  # 位置 i 被接受的次数
            assert isinstance(proposals_at_pos, list)
            assert isinstance(accepted_at_pos, list)
            accept_rates_by_position = []  # 逐位置接受率
            for pos_idx in range(self.max_proposal_tokens):
                position_proposal_count = proposals_at_pos[pos_idx]  # 该位置实际被提案过的次数
                if position_proposal_count == 0:  # 从未被提案到：None 占位
                    accept_rates_by_position.append(None)
                    continue
                accept_rates_by_position.append(  # 位置 i 接受率
                    accepted_at_pos[pos_idx] / position_proposal_count
                )
        return {  # 指标行结构
            "dataset": dataset_name,
            "num_samples": int(metric_summary["sample_count"]),  # 样本数
            "draft_tokens_per_proposal": draft_tokens_per_proposal,
            "acceptance_length": acceptance_length,
            "verify_rate": verify_rate,
            "accept_rates_by_position": accept_rates_by_position,
        }

    # 在本 rank 上跑完一个数据集的分片样本：从 global_rank 起步、以 world_size 为步长轮流取样本
    def run_dataset(
        self,
        *,
        dataset_name: str,  # 数据集名
        max_samples: int | None,  # 最大样本数上限；None 表示全量
    ) -> list[SimpleNamespace]:
        seed_all(int(self.args.seed))  # 先固定全局种子，保证下采样结果可复现
        dataset = load_and_process_dataset(dataset_name)  # 读取 JSONL 数据集

        if max_samples is not None and len(dataset) > max_samples:  # 超出上限：按种子随机下采样
            rng = random.Random(int(self.args.seed))  # 独立 RNG，不污染全局随机状态
            dataset = list(dataset)
            rng.shuffle(dataset)
            dataset = dataset[:max_samples]

        stop_token_ids = resolve_stop_token_ids(self.target_model, self.tokenizer)  # 解析 stop token
        responses = []
        for idx in range(self.global_rank, len(dataset), self.world_size):  # 数据并行：各 rank 间隔取样本
            seed_all(int(self.args.seed) + idx)  # 每条样本独立种子：结果与样本分配方式无关、可复现
            instance = dataset[idx]
            messages = [{"role": "user", "content": instance["turns"][0]}]  # 只用首轮 user 消息
            input_ids = encode_chat_messages(  # 套 chat template 编码为 token 序列
                self.tokenizer,
                messages,
                add_generation_prompt=True,  # 追加生成起始提示
                enable_thinking=False,
                # enable_thinking=True,
            ).to(self.device)
            responses.append(  # 逐样本生成
                self.generate_one_sample(
                    input_ids=input_ids,
                    stop_token_ids=stop_token_ids,
                )
            )

        return responses

    # 本 rank 逐样本累计投机解码计数，再跨 rank all_reduce 求全局总和。
    # 计数维度：验证轮数、每轮 accept length（含 bonus）、提案长度、逐位置提案/接受次数。
    def allreduce_response_metrics(
        self,
        responses: list[SimpleNamespace],  # 本 rank 的生成结果
    ) -> dict[str, int | list[int]]:
        metric_summary: dict[str, int | list[int]] = {  # 初始化汇总容器
            "sample_count": len(responses),  # 本 rank 样本数
            "proposal_count": 0,  # 验证总轮数
            "acceptance_length_sum": 0,  # 接受 token 总数（含 bonus）
            "proposal_length_sum": 0,  # 提案 token 总数
            "proposals_at_pos": [0] * self.max_proposal_tokens,  # 位置 i 被提案的次数
            "accepted_at_pos": [0] * self.max_proposal_tokens,  # 位置 i 被接受的次数
        }
        proposals_at_pos = metric_summary["proposals_at_pos"]  # 局部别名，便于循环内累加
        accepted_at_pos = metric_summary["accepted_at_pos"]
        assert isinstance(proposals_at_pos, list)
        assert isinstance(accepted_at_pos, list)

        for response in responses:
            acceptance_lengths = getattr(response, "acceptance_lengths", None)  # 每轮 accept length（含 bonus）
            proposal_lengths = getattr(response, "proposal_lengths", None)  # 每轮提案长度
            accepted_draft_lengths = getattr(response, "accepted_draft_lengths", None)  # 每轮接受 draft 数
            if (
                acceptance_lengths is None
                or proposal_lengths is None
                or accepted_draft_lengths is None
            ):
                continue  # 缺统计字段的结果（如非投机解码输出）跳过
            assert (
                len(acceptance_lengths)
                == len(proposal_lengths)
                == len(accepted_draft_lengths)
            ), (
                "acceptance_lengths, proposal_lengths and accepted_draft_lengths "
                "must have the same length."
            )
            for acceptance_length, proposal_length, accepted_draft_length in zip(
                acceptance_lengths,
                proposal_lengths,
                accepted_draft_lengths,
            ):
                metric_summary["proposal_count"] += 1  # 累计验证轮数
                metric_summary["acceptance_length_sum"] += int(acceptance_length)  # 累计接受 token
                metric_summary["proposal_length_sum"] += int(proposal_length)  # 累计提案长度
                accepted_draft_length = int(accepted_draft_length)
                for pos_idx in range(self.max_proposal_tokens):  # 逐位置计数
                    if proposal_length > pos_idx:  # 提案长度超过 i：位置 i 被提案过
                        proposals_at_pos[pos_idx] += 1
                    if accepted_draft_length > pos_idx:  # 接受数超过 i：位置 i 被接受过
                        accepted_at_pos[pos_idx] += 1

        scalar_tensor = torch.tensor(  # 4 个标量计数打包成一个张量，一次通信完成
            [
                int(metric_summary["sample_count"]),
                int(metric_summary["proposal_count"]),
                int(metric_summary["acceptance_length_sum"]),
                int(metric_summary["proposal_length_sum"]),
            ],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)  # 跨 rank 求和

        position_tensor = torch.tensor(  # 逐位置计数（提案 + 接受）拼接后一次通信
            proposals_at_pos + accepted_at_pos,
            device=self.device,
            dtype=torch.int64,
        )
        if position_tensor.numel() > 0:  # max_proposal_tokens 为 0 时跳过
            dist.all_reduce(position_tensor, op=dist.ReduceOp.SUM)
        return {  # 返回全局汇总
            "sample_count": int(scalar_tensor[0].item()),
            "proposal_count": int(scalar_tensor[1].item()),
            "acceptance_length_sum": int(scalar_tensor[2].item()),
            "proposal_length_sum": int(scalar_tensor[3].item()),
            "proposals_at_pos": position_tensor[
                : self.max_proposal_tokens
            ].tolist(),  # 前半段：逐位置提案计数
            "accepted_at_pos": position_tensor[
                self.max_proposal_tokens :
            ].tolist(),  # 后半段：逐位置接受计数
        }

    # 把各数据集指标写入 TensorBoard：以训练 step 为 x 轴，便于跟踪评测曲线变化
    def log_tensorboard(self) -> None:
        from torch.utils.tensorboard import SummaryWriter  # 延迟导入：仅在配置了输出目录时加载

        assert self.args.tensorboard_dir is not None
        assert self.args.step is not None  # 需要训练步数作为 global_step
        Path(self.args.tensorboard_dir).mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
        writer = SummaryWriter(log_dir=self.args.tensorboard_dir)
        for metrics in self.metrics_rows:
            dataset_name = metrics["dataset"]
            writer.add_scalar(  # 平均每轮提案长度
                f"eval/{dataset_name}/draft_tokens_per_proposal",
                float(metrics["draft_tokens_per_proposal"]),
                global_step=self.args.step,
            )
            writer.add_scalar(  # accept length（含 bonus）
                f"eval/{dataset_name}/accept_length",
                float(metrics["acceptance_length"]),
                global_step=self.args.step,
            )
            writer.add_scalar(  # 验证位置利用率
                f"eval/{dataset_name}/verify_rate",
                float(metrics["verify_rate"]),
                global_step=self.args.step,
            )
            for pos_idx, accept_rate in enumerate(metrics["accept_rates_by_position"]):
                if accept_rate is None:  # 从未被提案的位置跳过
                    continue
                writer.add_scalar(  # 逐位置接受率
                    f"eval/{dataset_name}/accept_rate@{pos_idx}",
                    float(accept_rate),
                    global_step=self.args.step,
                )
        writer.close()

    # rank 0 打印所有数据集的汇总表格
    def print_results(self) -> None:
        if dist.get_rank() != 0:  # 只在 rank 0 打印
            return
        print(
            build_results_table(
                rows=self.metrics_rows,
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
            ),
            flush=True,
        )

    # rank 0 打印单个数据集的结果行（无表头，便于逐数据集流水式查看）
    def print_dataset_result(self, metrics_row: dict[str, object]) -> None:
        if dist.get_rank() != 0:
            return
        print(
            build_results_table(
                rows=[metrics_row],
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
                header=False,  # 单行输出不带表头
            ),
            flush=True,
        )

    # 记录单数据集指标（rank 0 且有样本时）：构建指标行、缓存并立即打印；返回指标行
    def record_dataset_metrics(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | list[int]],
    ) -> dict[str, object] | None:
        if dist.get_rank() != 0 or int(metric_summary["sample_count"]) <= 0:  # 非 rank 0 或无样本不记录
            return None
        metrics_row = self.build_metrics_row(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
        )
        self.metrics_rows.append(metrics_row)  # 缓存供最终汇总
        self.print_dataset_result(metrics_row)  # 立即打印单数据集结果
        return metrics_row

    # 收尾上报：rank 0 在配置了目录且有指标时写 TensorBoard，随后打印汇总表格
    def report_results(self) -> None:
        if dist.get_rank() == 0 and self.metrics_rows:
            if self.args.tensorboard_dir is not None:
                self.log_tensorboard()
        self.print_results()

    # 评测主入口：逐任务（数据集）生成 -> 跨 rank 汇总 -> 记录，最后统一上报
    def evaluate(self) -> None:
        for dataset_name, max_samples in self.tasks:
            responses = self.run_dataset(  # 本 rank 生成各自分片的样本
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            metric_summary = self.allreduce_response_metrics(responses)  # 跨 rank 汇总指标
            self.record_dataset_metrics(  # rank 0 记录并打印
                dataset_name=dataset_name,
                metric_summary=metric_summary,
            )

        self.report_results()

    # 销毁分布式进程组，释放通信资源
    def clean_up(self) -> None:
        dist.destroy_process_group()
