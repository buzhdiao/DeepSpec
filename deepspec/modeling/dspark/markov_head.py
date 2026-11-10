# Markov 修正头族：给 draft 的基础 logits 注入低秩 token 转移偏置，实现块内半自回归——
# 块内并行预测后，前一（已采样/真值）token 通过偏置影响后一位置的分布，弥补纯并行缺失的依赖。
from typing import Optional

import torch  # PyTorch：张量操作、CUDA 设备属性与同步
from torch import nn

from deepspec.utils.sampling import sample_tokens  # 温度采样：temperature≈0 时退化为 argmax 贪心

class VanillaMarkov(nn.Module):
    # 无记忆低秩 Markov 头：把 V×V 的 token 转移偏置矩阵分解为两个低秩矩阵——
    # W1：词表 → R^r 的嵌入（前驱 token 的查询向量），W2：R^r → 词表的投影，
    # bias(x_prev) = W2 @ W1[x_prev]。只依赖前一个 token，推理时逐步施加偏置、训练时并行广播。
    def __init__(self, *, vocab_size: int, markov_rank: int):  # markov_rank=r：低秩瓶颈维，权衡表达力与参数量
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"  # 类型标签，便于 ckpt/日志区分
        assert self.markov_rank > 0, (
            f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        )  # 秩为 0 时 build_markov_head 直接返回 None，不该走到这里
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)  # W1[x]: [*, r] 前驱 token 的低秩嵌入
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)  # W2：r → vocab，无偏置

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())  # [*, r] 前驱 token 查嵌入（强制 long 以防 int32 输入）

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)  # [*, r] → [*, vocab]：低秩隐变量投影回词表维

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,  # 前驱 token id（上一已确定的 token）
        hidden_states: Optional[torch.Tensor],  # vanilla 不使用主干隐藏态，保留形参以统一各头接口
    ) -> torch.Tensor:
        del hidden_states  # 显式丢弃：vanilla 头无记忆，不看主干特征
        return self.project_bias(self.get_prev_embeddings(token_ids))  # bias = W2 @ W1[x_prev]

    def apply_step_logits(
        self,
        logits: torch.Tensor,  # [*, vocab] 待修正的 logits
        *,
        token_ids: torch.Tensor,  # 前驱 token id
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)  # 单步修正：logits 加上转移偏置

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,  # [batch, num_blocks, block_size, vocab] 训练期教师强制下的基础 logits
        *,
        token_ids: torch.Tensor,  # [batch, num_blocks, block_size] 每位前驱 token（来自真值标签）
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # 训练路径：teacher-forced 并行修正——前驱直接取真值标签，一次广播完成整块修正，无需展开循环。
        if base_logits.size(2) == 0:
            return base_logits  # 空块（block_size=0）直接返回
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias  # 广播加偏置：形状与 base_logits 相同

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,  # [batch, proposal_len, vocab] 推理期 draft 主干输出的块 logits
        *,
        first_prev_token_ids: torch.Tensor,  # [batch] 块首前驱 token（源序列 anchor 处 token）
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,  # 0 表示贪心 argmax
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 推理路径：块内逐步采样（半自回归）——第 k 步用第 k-1 步"实际采样出的 token"计算偏置并修正
        # logits，采样结果又作为下一步前驱，形成链式依赖。corrected_logits 返回逐步修正后的 logits，
        # 供目标模型做拒绝采样验证（训练/推理分布一致性依赖修正后的 logits）。
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )  # 空块占位输出
            return empty_tokens, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()  # 链首前驱：anchor token
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]  # 当前步的（可选）主干特征
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],
                token_ids=prev_token_ids,
                hidden_states=step_hidden,
            )  # 用前驱 token 修正本步 logits
            corrected_logits.append(step_logits.unsqueeze(1))  # 记录 [batch, 1, vocab]
            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)  # 采样本步 token（temperature≈0 时 argmax）
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids  # 关键：前驱更新为采样结果（半自回归链推进）
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)  # ([batch, L], [batch, L, vocab])


class GatedMarkovHead(VanillaMarkov):
    # 门控 Markov 头：在 vanilla 之上加门控——用 draft 主干隐藏态 h_k 与前驱嵌入拼接出 [0,1] 门，
    # 对低秩隐变量逐维调制后再投影，让"token 转移先验"的强度由当前语义状态动态决定。
    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,  # 主干隐藏维：参与门控输入
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"  # 类型标签：gated
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)  # [h_k; W1[x_prev]] → r 个门 logit

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],  # 必须提供：门控依赖主干隐藏态
    ) -> torch.Tensor:
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)  # 拼接主干特征与前驱嵌入
        return torch.sigmoid(self.gate_proj(gate_inputs))  # [*, r] ∈ (0,1)：逐维门控系数

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(dtype=prev_embeddings.dtype)  # 门控与嵌入对齐 dtype（fp16/bf16 训练）
        return self.project_bias(gate * prev_embeddings)  # bias = W2 @ (g ⊙ W1[x_prev])：门控调制后的转移偏置


class RNNHead(VanillaMarkov):
    """RNN-based head that maintains recurrent state across positions within a block.

    Unlike the memoryless Markov heads, position k can access the full prefix
    history x_{<k} through a GRU-like recurrent state.
    """
    # 循环神经网络头：块内维护 GRU 风格递归状态 s_k，使位置 k 能看到全部前缀历史 x_{<k}，
    # 突破 vanilla/gated 只依赖前一 token 的记忆限制。
    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"  # 类型标签：rnn
        self.hidden_size = hidden_size
        # Joint projection: [s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output]
        # W_g, W_c, W_o in R^{(2r+d) x r} parameterized as one linear layer.
        # （原注释）三个投影合一：一次矩阵乘同时产出门/候选/输出三组 r 维激活，省内核启动开销
        self.state_size = markov_rank  # 递归状态维 = r
        self.joint_proj = nn.Linear(
            2 * markov_rank + hidden_size, 3 * markov_rank
        )  # 输入 [s_{k-1}(r); W1[x_{k-1}](r); h_k(d)]，输出 [gate; candidate; output](3r)

    def _rnn_step(
        self,
        state: torch.Tensor,  # [*, r] 上一步递归状态
        prev_embeddings: torch.Tensor,  # [*, r] W1[x_{k-1}]
        hidden_states: torch.Tensor,  # [*, d] 第 k 步主干隐藏态
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single RNN step.

        Args:
            state: [*, r] previous recurrent state
            prev_embeddings: [*, r] W1[x_{k-1}]
            hidden_states: [*, d] backbone hidden at step k

        Returns:
            new_state: [*, r]
            bias: [*, vocab_size]
        """
        # 单步递归：门控凸组合更新状态，输出支路独立产生偏置（GRU 变体：省去重置门）。
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)  # [*, 2r+d] 拼接三路输入
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)  # 拆成三路激活
        gate = torch.sigmoid(gate_raw)  # 更新门 g ∈ (0,1)
        candidate = torch.tanh(candidate_raw)  # 候选状态
        new_state = gate * state + (1.0 - gate) * candidate  # 凸组合：g 越大越保留旧状态
        bias = self.project_bias(torch.tanh(output_raw))  # 输出支路 → 词表维偏置
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Stateless single-step bias (state initialized to zero).

        This is used for compatibility but does NOT carry state across steps.
        For full RNN behavior, use apply_block_logits or sample_block_tokens.
        """
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)  # 零初始状态：不带任何前缀信息
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias  # 仅单步偏置，供接口兼容（真正的递归在 block 级方法里）

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply RNN bias during training (teacher-forced, unrolled over block_size).

        Args:
            base_logits: [B, num_blocks, block_size, V]
            token_ids: [B, num_blocks, block_size] - prev token ids for each position
            hidden_states: [B, num_blocks, block_size, d]
        """
        # 训练路径：teacher-forced 沿块内时间展开递归——状态在块内逐步推进、块间独立重置。
        assert hidden_states is not None
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits  # 空块直通

        leading_shape = base_logits.shape[:-2]  # e.g. [B, num_blocks]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )  # 每块各自独立、零初始化的递归状态

        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])  # 第 k 位前驱（teacher-forced 真值）
            h_k = hidden_states[..., k, :]  # 第 k 位主干隐藏态
            state, bias = self._rnn_step(state, prev_emb, h_k)  # 递归状态沿块内时间推进
            output_logits.append(base_logits[..., k, :] + bias)  # 修正本位 logits

        return torch.stack(output_logits, dim=-2)  # 还原 [.., block_size, vocab]

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive sampling with RNN state.

        Args:
            base_logits: [batch, proposal_len, vocab]
            first_prev_token_ids: [batch] - token preceding the first draft position
            hidden_states: [batch, proposal_len, d]
            temperature: sampling temperature

        Returns:
            sampled_tokens: [batch, proposal_len]
            corrected_logits: [batch, proposal_len, vocab]
        """
        # 推理路径：带递归状态的逐步采样——前驱取上一步采样结果，状态跨步传递（完整 RNN 行为）。
        assert hidden_states is not None
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )  # 空块占位输出
            return empty_tokens, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )  # 推理起始零状态

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()

        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_token_ids)  # 前驱取上一步实际采样 token
            h_k = hidden_states[:, step_idx, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)  # 状态跨步传递

            step_logits = base_logits[:, step_idx, :] + bias  # 修正本步 logits
            corrected_logits.append(step_logits.unsqueeze(1))

            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)  # 温度采样（temperature≈0 时 argmax）
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids  # 半自回归链推进

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)  # ([batch, L], [batch, L, vocab])


def build_markov_head(config) -> nn.Module | None:
    # 修正头工厂：markov_rank == 0 → None（无修正头，块内退化为纯并行预测）；
    # 否则按 markov_head_type 实例化 vanilla / gated / rnn 三种头之一。
    markov_rank = int(config.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None  # 显式关闭修正头

    markov_head_type = str(config.markov_head_type).lower()  # 类型名大小写不敏感
    if markov_head_type == "vanilla":
        return VanillaMarkov(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    assert False, f"Unsupported markov_head_type: {markov_head_type!r}"  # 未知类型直接失败，防止静默用错头


__all__ = [
    "VanillaMarkov",
    "GatedMarkovHead",
    "RNNHead",
    "build_markov_head",
]
