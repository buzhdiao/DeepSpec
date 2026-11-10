# Gemma4 Eagle3 draft 配置构造：从目标模型的 text_config 派生 draft 配置，
# 注入抽层、TTT 训练参数与注意力实现等 draft 专属字段。
import copy  # 深拷贝：draft 配置基于目标配置副本修改，不污染原对象

from deepspec.modeling.eagle3.common import validate_eagle3_target_layer_ids  # 目标层号校验（固定 5 层、递增、在界内）


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"  # draft 训练注意力实现固定为 flex_attention（Eagle3 TTT 掩码需 BlockMask）


def get_gemma4_text_config(target_config):
    # 从 Gemma4 顶层（可为多模态 unified 包装）配置中取出文本子配置的深拷贝。
    # draft 与目标文本模型同词表/同维度，直接以 text_config 为底座改写字段。
    assert target_config.model_type in ("gemma4", "gemma4_unified"), (
        "Gemma4 Eagle3 expects a Gemma4 or Gemma4 Unified top-level target config, "
        f"got model_type={target_config.model_type!r}."
    )
    text_config = target_config.text_config  # 多模态包装下的文本子配置（纯文本模型亦有该字段）
    assert text_config.model_type in ("gemma4_text", "gemma4_unified_text"), (
        "Gemma4 Eagle3 expects target_config.text_config.model_type to be "
        f"'gemma4_text' or 'gemma4_unified_text', got {text_config.model_type!r}."
    )
    return copy.deepcopy(text_config)  # 深拷贝，避免改写共享的目标配置


def _validate_required_text_fields(text_config) -> None:
    # 前置校验 draft modeling 直接读取的全部 text_config 字段：
    # 缺字段若拖到模型初始化才报错，排查成本高；这里集中 fail-fast 并给出明确信息。
    required_fields = (
        "vocab_size",  # 词表大小（draft 头与目标共享）
        "hidden_size",  # 隐层宽度
        "intermediate_size",  # MLP 中间层宽度
        "num_hidden_layers",  # 目标模型层数（据此校验抽层范围）
        "num_attention_heads",  # 注意力 query 头数
        "num_global_key_value_heads",  # 全局注意力 KV 头数（attention_k_eq_v 时使用）
        "global_head_dim",  # 全局注意力头维（draft 的 head_dim）
        "attention_bias",  # q/k/v/o 投影是否带 bias
        "attention_dropout",  # 注意力 dropout
        "attention_k_eq_v",  # K=V 注意力变体开关（全局层复用 k 作 v）
        "enable_moe_block",  # MoE 块开关（draft 原型不支持，需为 False）
        "head_dim",  # 注意力头维
        "hidden_activation",  # 激活函数类型
        "hidden_size_per_layer_input",  # Gemma4 逐层输入宽度（draft 不支持，需为 0）
        "initializer_range",  # 权重初始化幅度
        "max_position_embeddings",  # 最大位置数
        "num_key_value_heads",  # 局部注意力 KV 头数
        "num_kv_shared_layers",  # KV 共享层配置
        "rms_norm_eps",  # RMSNorm epsilon
        "rope_parameters",  # RoPE 参数（含全局/局部层的 theta 设置）
        "use_double_wide_mlp",  # 双宽 MLP 开关
    )
    for field in required_fields:
        assert hasattr(text_config, field), (
            f"target_config.text_config.{field} must be provided."
        )


def build_draft_config(*, target_config, model_args):
    # 构造 draft 配置：以目标 text_config 深拷贝为底座，校验抽层与 TTT 超参后，
    # 写入 draft 层数与各 Eagle3 专属字段。返回值供 Gemma4Eagle3Model 初始化与保存。
    draft_config = get_gemma4_text_config(target_config)
    _validate_required_text_fields(draft_config)

    num_target_layers = int(draft_config.num_hidden_layers)  # 目标模型层数（抽层校验的上界）
    target_layer_ids = validate_eagle3_target_layer_ids(
        model_args.target_layer_ids,  # 从目标模型抽取的层号（固定 5 个）
        num_target_layers,
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

    draft_config.architectures = ["Gemma4Eagle3Model"]  # HF 自动加载时映射到的 draft 模型类
    draft_config.target_model_type = str(target_config.model_type)  # 记录目标顶层类型（溯源）
    draft_config.target_text_model_type = str(draft_config.model_type)  # 记录目标文本模型类型（溯源）
    draft_config.num_target_layers = num_target_layers  # 目标层数（供运行时校验/展示）
    draft_config.num_hidden_layers = draft_num_hidden_layers  # draft 层数覆盖为 draft 深度
    draft_config.layer_types = ["full_attention"] * draft_num_hidden_layers  # draft 全部用全注意力（不做滑动窗口混合）
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
    "get_gemma4_text_config",
]
