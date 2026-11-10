# DSpark draft 模型包入口（DSpark：置信度调度 + 半自回归生成的投机解码草稿模型）。
# 导出结构：共享的前向输出容器与目标特征抽取函数，以及 Gemma4/Qwen3 两种骨干的 draft 实现。
from .common import DSparkForwardOutput, extract_context_feature  # 一次训练前向的输出容器 + 按目标层 id 拼接目标隐藏态特征
from .gemma4 import Gemma4DSparkModel  # Gemma4 骨干的 DSpark draft 模型（跨注意力注入目标模型隐藏态）
from .qwen3 import Qwen3DSparkModel  # Qwen3 骨干的 DSpark draft 模型（与 Gemma4 版同构，注意力实现不同）

__all__ = [  # from deepspec.modeling.dspark import * 时的可见名字清单
    "DSparkForwardOutput",  # draft_logits/target_ids/eval_mask/block_keep_mask/confidence_pred 等训练产物
    "extract_context_feature",  # 从目标模型各层隐藏态抽取并拼接 draft 的跨注意上下文特征
    "Gemma4DSparkModel",  # 供 trainer/eval 按架构名实例化的 Gemma4 版 draft 模型
    "Qwen3DSparkModel",  # 供 trainer/eval 按架构名实例化的 Qwen3 版 draft 模型
]
