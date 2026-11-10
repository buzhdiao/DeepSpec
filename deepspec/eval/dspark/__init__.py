# dSpark 评测子包出口：评测器 + 整块提案构造工具 + 置信度头校准统计
from .evaluator import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator  # 两种目标模型家族的 dSpark 评测器
from .draft_ops import (  # dSpark 提案相关工具（一次前向产出整块 token 的核心逻辑）
    DSparkDraftProposal,  # 扩展 DraftProposal：附加置信度 logits 的提案
    build_dspark_proposal,  # 采样整块 draft token 并按置信度阈值截断，构造提案
    forward_dspark_draft_block,  # 跑一遍 draft backbone，并行产出整块隐状态
)
from .confidence_head import ConfidenceHeadRecorder  # 置信度头校准指标（ECE/AUROC/Brier）记录器

__all__ = [  # 公开导出清单
    "Gemma4DSparkEvaluator",  # Gemma4 版 dSpark 评测器
    "Qwen3DSparkEvaluator",  # Qwen3 版 dSpark 评测器
    "DSparkDraftProposal",  # 带置信度 logits 的提案数据类
    "build_dspark_proposal",  # 提案构造函数
    "forward_dspark_draft_block",  # draft 块前向函数
    "ConfidenceHeadRecorder",  # 置信度校准记录器
]
