# deepspec.trainer 子包导出层：按"算法家族 × 模型家族"组合暴露四个具体训练器。
# 基类 BaseTrainer 负责分布式/FSDP/数据/断点续训等通用流程；
# 子类只注入两件事：构建哪种 draft 模型（_build_draft_model）与单步损失计算（run_batch）。
from .base_trainer import BaseTrainer  # 通用训练基类：FSDP 封装、梯度累积训练循环、checkpoint/断点续训
from .dspark_trainer import Gemma4DSparkTrainer, Qwen3DSparkTrainer  # DSpark 算法训练器（两模型家族）
from .eagle3_trainer import Gemma4Eagle3Trainer, Qwen3Eagle3Trainer  # EAGLE3 算法训练器（两模型家族）

__all__ = [  # 公开符号：一个基类 + 四个具体训练器（2 算法 × 2 模型家族）
    "BaseTrainer",  # 训练器基类
    "Gemma4Eagle3Trainer",  # EAGLE3 + Gemma4 draft
    "Gemma4DSparkTrainer",  # DSpark + Gemma4 draft
    "Qwen3Eagle3Trainer",  # EAGLE3 + Qwen3 draft
    "Qwen3DSparkTrainer",  # DSpark + Qwen3 draft
]
