# Qwen3 Eagle3 draft 配置构造：从目标模型配置深拷贝派生 draft 配置，
# 注入抽层、TTT 训练参数与注意力实现等 draft 专属字段。
import copy  # 深拷贝：draft 配置基于目标配置副本修改，不污染原对象

from deepspec.modeling.eagle3.common import validate_eagle3_target_layer_ids  # 目标层号校验（固定 5 层、递增、在界内）


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"  # draft 训练注意力实现固定为 flex_attention（Eagle3 TTT 掩码需 BlockMask）


def build_draft_config(*, target_config, model_args):
    # 构造 draft 配置：Qwen3 为纯文本模型，直接深拷贝目标配置；校验抽层与 TTT 超参后
    # 覆写 draft 层数与各 Eagle3 专属字段。返回值供 Qwen3Eagle3Model 初始化与保存。
    target_layer_ids = validate_eagle3_target_layer_ids(
        model_args.target_layer_ids,  # 从目标模型抽取的层号（固定 5 个）
        int(target_config.num_hidden_layers),
    )
    ttt_length = int(model_args.ttt_length)  # TTT 步数：每步模拟一次自回归起草
    assert ttt_length >= 1, f"ttt_length must be >= 1, got {ttt_length}"
    step_loss_decay = float(model_args.step_loss_decay)  # 步间损失衰减（第 s 步权重 decay^s）
    assert step_loss_decay > 0.0, (
        "step_loss_decay must be > 0.0, "
        f"got {step_loss_decay}"
    )
    draft_num_hidden_layers = int(model_args.draft_num_hidden_layers)  # draft 自身解码层层数
    assert draft_num_hidden_layers >= 1, (
        "draft_num_hidden_layers must be >= 1, "
        f"got {draft_num_hidden_layers}"
    )

    draft_config = copy.deepcopy(target_config)  # 与目标同词表/同维度，深拷贝后改写
    draft_config.architectures = ["Qwen3Eagle3Model"]  # HF 自动加载时映射到的 draft 模型类
    draft_config.num_target_layers = int(target_config.num_hidden_layers)  # 目标层数（供运行时校验/展示）
    draft_config.num_hidden_layers = draft_num_hidden_layers  # draft 层数覆盖为 draft 深度
    draft_config.layer_types = ["full_attention"] * draft_num_hidden_layers  # draft 全部用全注意力
    draft_config.target_model_name_or_path = str(model_args.target_model_name_or_path)  # 目标权重路径（embedding/head 来源）
    draft_config.target_layer_ids = target_layer_ids  # 抽层号（5 层，特征拼接顺序）
    draft_config.ttt_length = ttt_length
    draft_config.step_loss_decay = step_loss_decay
    draft_config.draft_num_hidden_layers = draft_num_hidden_layers
    draft_config.tie_word_embeddings = False  # draft 的 lm_head 独立持有目标头权重（不与 embedding 绑定）
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION  # 训练注意力实现：flex_attention
    return draft_config


__all__ = [
    "build_draft_config",
]
