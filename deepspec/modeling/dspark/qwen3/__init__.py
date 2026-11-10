# Qwen3 版 DSpark draft 子包：配置工厂 + 模型实现。
from .config import build_draft_config  # 由目标 Qwen3 配置派生 draft 配置（层数/块大小/mask token/Markov 与置信度头开关）
from .modeling import Qwen3DSparkModel  # DSpark draft 模型主体（Qwen3 架构：跨注意吃目标隐藏态 + 块内半自回归）

__all__ = [
    "Qwen3DSparkModel",  # HF 按 architectures=["Qwen3DSparkModel"] 反射加载的模型类
    "build_draft_config",  # 训练/加载前由 target_config + model_args 组装 draft config 的工厂
]
