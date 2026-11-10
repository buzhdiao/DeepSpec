# EAGLE3 评测子包出口：导出两种目标模型家族的评测器实现
from .evaluator import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator  # 转发导出（两套仅 draft 模型类不同，逻辑同在 evaluator.py）

__all__ = ["Gemma4Eagle3Evaluator", "Qwen3Eagle3Evaluator"]  # 公开导出清单
