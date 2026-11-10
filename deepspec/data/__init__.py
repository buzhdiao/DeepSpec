# deepspec.data 子包导出层：集中暴露聊天模板注册表与目标缓存（target cache）数据管线。
# 目标缓存把目标模型（teacher）的隐状态离线物化到磁盘，draft model 训练时无需再跑目标模型前向。
from .parser import TEMPLATE_REGISTRY  # 聊天模板注册表：按名称索引各模型家族的对话渲染模板
from .target_cache_dataset import (  # 目标缓存协议实现：写入端（writer）、读取端（dataset）与整理器
    CacheCollator,  # 目标缓存样本批整理器：按批内最长序列补零，并从 input_ids 重建 attention_mask
    CacheDataset,  # 目标缓存数据集：mmap 只读 shard 二进制 + samples.idx 定长索引记录定位样本
    ConversationCollator,  # 原始 JSONL 对话样本整理器：现场 tokenize/构造 loss_mask，过滤无效样本
    validate_train_cache,  # 校验缓存元信息与 draft 模型结构、target 模型名完全匹配，防止错配
)

__all__ = [  # `from deepspec.data import *` 与 IDE 自动补全可见的公开符号列表
    "CacheCollator",  # 缓存样本 → padding 后的训练 batch
    "CacheDataset",  # 目标缓存只读数据集
    "ConversationCollator",  # 对话样本 → (input_ids, attention_mask, loss_mask) batch
    "TEMPLATE_REGISTRY",  # ChatTemplate 注册表实例
    "validate_train_cache",  # 缓存与模型配置一致性断言
]
