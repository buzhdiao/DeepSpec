# 评测模块统一出口：聚合投机解码评测的基础框架与两类 draft 实现的评测器
from .base_evaluator import BaseEvaluator, DraftProposal, VerificationResult  # 基础评测器、draft 提案与验证结果的数据结构
from .dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator  # dSpark 评测器：整块并行提案 + 可选置信度截断
from .eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator  # EAGLE3 评测器：自回归 draft + KV cache 复用

__all__ = [  # 公开导出清单：from deepspec.eval import * 时可见的名字
    "BaseEvaluator",  # 评测器基类：数据加载、生成循环、指标统计与分布式汇总
    "DraftProposal",  # draft 提案数据类（待验证 token 与 draft 概率）
    "Gemma4Eagle3Evaluator",  # Gemma4 目标模型的 EAGLE3 评测器
    "Gemma4DSparkEvaluator",  # Gemma4 目标模型的 dSpark 评测器
    "Qwen3Eagle3Evaluator",  # Qwen3 目标模型的 EAGLE3 评测器
    "Qwen3DSparkEvaluator",  # Qwen3 目标模型的 dSpark 评测器
    "VerificationResult",  # target 验证结果数据类（接受数、bonus token 等）
]
