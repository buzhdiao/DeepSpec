# Eagle3 训练配置（Qwen3-8B）：TTT（test-time training）式草稿训练——训练时把
# 草稿模型自回归展开 ttt_length=7 步逐 token 预测未来；与 4B 版的差异仅在于目标
# 模型为 8B（同为 36 层、相同目标层取点）。
import os  # 路径拼接（finalize_cfg 中拼输出目录）

from deepspec.trainer import Qwen3Eagle3Trainer  # Qwen3 版 Eagle3 训练器
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_8B  # 公共常量：输出根目录与目标模型路径


project_name = "deepspec"  # 项目名：checkpoint/TensorBoard 目录第一级
exp_name = "eagle3_ttt7_qwen3_8b"  # 实验名：算法 + 展开步数 ttt7 + 目标模型，目录第二级
seed = 0  # 全局随机种子（eagle3 系列统一 0；dflash/dspark 系列为 42）

model = dict(  # 模型结构超参
    target_model_name_or_path=QWEN_3_8B,  # 目标模型（被加速的大模型）HF 权重路径
    target_layer_ids=[1, 9, 17, 25, 33],  # 拼接进草稿输入的目标层特征（Eagle3 固定取 5 层且须严格递增；Qwen3-8B 共 36 层）
    ttt_length=7,  # 训练时草稿模型自回归展开步数：逐步预测 7 个未来 token（exp_name 中 ttt7 即由此）
    step_loss_decay=0.8,  # 第 k 步损失权重 = 0.8^k：早期步（最影响实际推理接受率）权重更高
    draft_num_hidden_layers=1,  # 草稿模型仅 1 层 transformer：极轻量，推理开销最小
)

train = dict(  # 训练超参
    trainer_cls=Qwen3Eagle3Trainer,  # 训练器类（配置内保留类对象，dump 日志时记类名）
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
    torch_compile=False,  # 关闭 torch.compile（dflash/dspark 的 Qwen3 配置为 True）
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
    project_name=str(cfg['project_name'])  # 项目名（保留原有紧凑写法）
    exp_name = str(cfg["exp_name"])  # 实验名
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)  # checkpoint 输出目录：{根}/{项目}/{实验}
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)  # TensorBoard 日志目录：同上结构
    cfg["logging"] = logging_cfg  # 写回配置
    
    return cfg
