# 采样工具集：投机推理中目标/草稿模型概率分布的构建、按温度采样与残差采样。
from __future__ import annotations  # 允许注解使用前向引用（PEP 563 语义）

import torch  # 张量运算


# logits → 概率分布。temperature < 1e-5 视为贪心：直接构造 one-hot（argmax 置 1.0），
# 避免除以极小温度导致的数值溢出；常规分支在 fp32 内做 softmax 保证稳定。
def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:  # logits：[batch, seq, vocab]；temperature：采样温度
    if temperature < 1e-5:  # 贪心分支
        probs = torch.zeros_like(logits, dtype=torch.float32)  # 全 0 的 fp32 概率底板
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)  # argmax 位置置 1，其余 0
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)  # fp32 除温度再 softmax（低温也安全）


# 从 [batch, seq, vocab] 概率分布逐位置采一个 token，返回 [batch, seq] 的 token id。
def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, vocab_size = probs.shape  # 展开三维形状
    flat = probs.reshape(-1, vocab_size)  # 摊平成二维：multinomial 按行独立采样
    return torch.multinomial(flat, num_samples=1).reshape(bsz, seq_len)  # 每行按概率采 1 个后恢复形状


# 顶层采样入口：低温走贪心（确定性、评测可复现），否则按温度缩放后多项式采样。
def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:  # 默认温度 0 即贪心
    if temperature < 1e-5:  # 贪心
        return torch.argmax(logits, dim=-1)  # 直接取最大 logit 下标

    bsz, seq_len, vocab_size = logits.shape  # 展开形状
    flat_logits = logits.reshape(-1, vocab_size) / temperature  # 摊平并按温度缩放（越大越随机）
    probs = torch.softmax(flat_logits, dim=-1)  # 转概率分布
    return torch.multinomial(probs, num_samples=1).reshape(bsz, seq_len)  # 采样并恢复形状


# 取指定 token 的概率：[batch, seq, vocab] + [batch, seq] → [batch, seq]；
# 用于投机解码中查询草稿/目标模型对同一 token 的概率以计算接受率。
def gather_token_probs(probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)  # 最后一维按下标 gather 后挤掉该维


# 残差采样：标准投机采样的"补采"步骤——接受判定后，剩余概率质量
# clamp(target-draft, min=0) 构成补采分布，从中采出修正 token；
# 草稿完全覆盖目标（残差质量≈0）的位置退回 target 分布；除法 clamp_min(1e-8) 防零。
def sample_residual(
    target_probs: torch.Tensor,  # 目标模型概率 [batch, vocab]（单个位置）
    draft_probs: torch.Tensor,  # 草稿模型概率（同形状，与 target 同一位置）
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)  # 残差分布：负值截 0（和未必为 1）
    residual_mass = residual.sum(dim=-1, keepdim=True)  # 每行的残差总质量
    if torch.any(residual_mass <= 1e-8):  # 存在草稿完全覆盖目标的位置
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)  # 这些位置退回 target 分布
        residual_mass = residual.sum(dim=-1, keepdim=True)  # 重算质量（此时为 1）
    residual = residual / residual_mass.clamp_min(1e-8)  # 归一化为合法分布
    return sample_from_probs(residual.unsqueeze(1)).squeeze(1)  # 插入 seq 维借用三维接口采样后挤掉


__all__ = [  # 对外导出的采样 API
    "gather_token_probs",  # 取指定 token 的概率
    "logits_to_probs",  # logits 转概率（含贪心 one-hot）
    "sample_from_probs",  # 从概率分布采样
    "sample_residual",  # 投机采样残差补采
    "sample_tokens",  # 按温度采样 token
]
