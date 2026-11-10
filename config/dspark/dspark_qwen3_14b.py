# DSpark 训练配置（Qwen3-14B）：块级草稿模型的完整形态——markov 头 + 置信度头 +
# L1 蒸馏为主的损失。与 Gemma-4-12B 版的差异：目标层取点、mask token、开启 torch.compile。
import os  # 路径拼接（finalize_cfg 中拼输出目录）

from deepspec.trainer import Qwen3DSparkTrainer  # Qwen3 版 DSpark 训练器
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_14B  # 公共常量：输出根目录与目标模型路径


project_name = "deepspec"  # 项目名：checkpoint/TensorBoard 目录第一级
exp_name = "dspark_block7_qwen3_14b"  # 实验名：算法 + 块长 7 + 目标模型，目录第二级
seed = 42  # 全局随机种子（dflash/dspark 系列统一 42；eagle3 系列为 0）

model = dict(  # 模型结构超参
    target_model_name_or_path=QWEN_3_14B,  # 目标模型（被加速的大模型）HF 权重路径
    block_size=7,  # 草稿块长度：一次并行草稿 7 个 token（exp_name 中 block7 即由此）
    num_draft_layers=5,  # 草稿模型层数：5 层轻量 transformer
    target_layer_ids=[1, 10, 19, 28, 37],  # 抽取目标模型特征的层号：浅/中/深混合（Qwen3-14B 共 40 层）
    mask_token_id=151669,  # Qwen3 词表专用 mask token：锚点块内被遮蔽位置占位（Gemma 系列为 4）
    num_anchors=512,  # 每条样本采样的锚点块数：每次前向同时预测 512 个块

    # markov head
    markov_rank=256,  # markov 头低秩维度：词表↔256 维低秩分解，建模"上一块真实 token → 下一块"的一阶依赖（DFlash 置 0 关闭）
    markov_head_type="vanilla",  # markov 头结构：vanilla 低秩双线性（另有 gated/rnn 变体）

    # confidence head
    confidence_head_alpha=1.0,  # 置信度损失权重：1.0 与 CE/L1 联合训练块级置信度打分（DFlash 置 0 关闭）
    confidence_head_with_markov=True,  # 置信度头以 markov 头特征为输入：两者协同

    # loss
    loss_decay_gamma=4.0,  # 块内逐位置损失指数衰减 exp(-pos/gamma)：块首权重高（gamma=4 缓衰减）
    ce_loss_alpha=0.1,  # 交叉熵损失权重：辅助项
    l1_loss_alpha=0.9,  # logits L1 蒸馏损失权重：主项，直接对齐目标分布
)

train = dict(  # 训练超参
    trainer_cls=Qwen3DSparkTrainer,  # 训练器类（配置内保留类对象，dump 日志时记类名）
    lr=6.0e-4,  # 峰值学习率
    warmup_ratio=0.04,  # 线性预热占总步数 4%
    weight_decay=0.0,  # 权重衰减：关闭
    precision="bf16",  # bf16 混合精度训练
    local_batch_size=1,  # 每卡 micro-batch：目标 hidden states 缓存数据大，1 起步
    global_batch_size=512,  # 全局 batch：跨卡聚合 512 条
    num_train_epochs=10,  # 数据遍历 10 轮
    max_train_steps=None,  # 步数上限：None 表示由 epoch 数推算
    max_grad_norm=1.0,  # 梯度范数裁剪阈值
    sharding_strategy="no_shard",  # FSDP 不分片：草稿模型很小，无需切分
    torch_compile=True,  # 开启 torch.compile 加速（Gemma4 路径为 False）
)

logging = dict(  # 日志与存档
    logging_steps=10,  # 每 10 个 optimizer step 打印/写 TB 一次
    checkpointing_steps=3000,  # 每 3000 步存一次 checkpoint
)

data = dict(  # 数据管线
    target_cache_path=None,  # 预计算目标模型 hidden states 缓存路径：None 走默认缓存目录
    chat_template="qwen",  # 对话模板：与目标模型匹配（Gemma 系列为 "gemma4"）
    max_length=4096,  # 序列最大长度
    num_workers=4,  # dataloader worker 数
)


# 配置后处理钩子（load_config 后由 finalize_config 调用）：把 checkpoint 与
# TensorBoard 目录按 {基目录}/{项目名}/{实验名} 拼好写回 logging 段。
def finalize_cfg(cfg):  # cfg：完整配置字典
    logging_cfg = dict(cfg["logging"])  # 拷贝 logging 段以便就地补字段
    project_name = str(cfg["project_name"])  # 项目名
    exp_name = str(cfg["exp_name"])  # 实验名
    logging_cfg["checkpoint_dir"] = os.path.join(  # checkpoint 输出目录
        BASE_CKPT_DIR,  # 根目录 ~/checkpoints
        project_name,  # 项目级子目录
        exp_name,  # 实验级子目录
    )
    logging_cfg["tensorboard_dir"] = os.path.join(  # TensorBoard 日志目录
        BASE_TB_DIR,  # 根目录 ~/tensorboard
        project_name,
        exp_name,
    )
    cfg["logging"] = logging_cfg  # 写回配置
    return cfg
