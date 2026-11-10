# Qwen3 目标模型对应的 Eagle3 draft 子包入口。
# draft 与目标同词表/同词嵌入/同输出头，主体为"低秩投影 fc + 若干 draft 解码层"。
from .config import build_draft_config  # 由目标模型配置派生 draft 配置（抽层/TTT 参数/注意力实现）
from .modeling import Qwen3Eagle3Model  # Qwen3 版 Eagle3 draft 模型实现

__all__ = [
    "Qwen3Eagle3Model",  # draft 模型本体
    "build_draft_config",  # draft 配置构造入口
]
