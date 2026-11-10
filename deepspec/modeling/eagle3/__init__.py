# DeepSpec Eagle3 投机解码 draft 模型包的入口。
# Eagle3（EAGLE-3）用目标模型多层中间隐状态（低秩投影压缩）+ token embedding 作为
# draft 输入，以 TTT（训练时模拟多步自回归起草）方式蒸馏 draft 头。本包包含：
#   - common.py ：三源上下文特征提取、Eagle3 TTT FlexAttention 掩码、前向输出容器等公共件
#   - loss.py   ：Triton 融合的 soft 交叉熵 TTT 蒸馏损失
#   - qwen3/    ：Qwen3 目标模型对应的 draft 实现
#   - gemma4/   ：Gemma4 目标模型对应的 draft 实现
from .common import Eagle3ForwardOutput, extract_eagle3_context_feature  # 前向输出容器与多层特征拼接工具
from .gemma4 import Gemma4Eagle3Model  # Gemma4 版 Eagle3 draft 模型
from .qwen3 import Qwen3Eagle3Model  # Qwen3 版 Eagle3 draft 模型

__all__ = [  # 公开导出：训练/评测脚本仅从这里引用 Eagle3 组件
    "Eagle3ForwardOutput",  # draft 单次前向的输出（隐状态 + draft logits + 可选目标 logits）
    "Gemma4Eagle3Model",  # Gemma4 draft 模型（低秩投影 fc + draft 解码层 + 目标词表头）
    "Qwen3Eagle3Model",  # Qwen3 draft 模型（结构与上者同构）
    "extract_eagle3_context_feature",  # 把目标模型各指定层隐状态沿特征维拼成 draft 输入特征
]
