# Gemma4 版 DSpark draft 配置构造：从（可能是多模态 Unified 的）目标配置取出文本配置，再派生 draft 配置。
import copy  # 深拷贝配置：draft 与 target 各持一份文本配置对象，避免共享可变状态

from deepspec.modeling.dspark.common import validate_target_layer_ids  # 校验 target_layer_ids：非空、严格递增、范围合法（-1 表示 embedding 输出）


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"  # 训练期注意力实现固定为 flex_attention，以使用 create_block_mask 的块稀疏掩码


def get_gemma4_text_config(target_config):
    # Gemma4 存在多模态顶层配置（gemma4 / gemma4_unified）：draft 只需要其中的文本塔 text_config。
    # 校验顶层与文本子配置的 model_type 后，深拷贝文本配置返回（后续改写字段不污染目标配置）。
    assert target_config.model_type in ("gemma4", "gemma4_unified"), (
        "Gemma4 DSpark expects a Gemma4 or Gemma4 Unified top-level target config, "
        f"got model_type={target_config.model_type!r}."
    )  # 顶层必须是 Gemma4 家族
    text_config = target_config.text_config  # 取出文本子配置（多模态模型中的语言塔）
    assert text_config.model_type in ("gemma4_text", "gemma4_unified_text"), (
        "Gemma4 DSpark expects target_config.text_config.model_type to be "
        f"'gemma4_text' or 'gemma4_unified_text', got {text_config.model_type!r}."
    )  # 文本子配置类型也须匹配
    return copy.deepcopy(text_config)  # 深拷贝副本作为 draft 配置基底


def _validate_required_text_fields(text_config) -> None:
    # 校验文本配置包含 draft 构建所需的全部 Gemma4 字段，缺失即尽早失败（避免训练中途才报错）。
    required_fields = (
        "vocab_size",  # 词表大小：embed/lm_head 形状来源
        "hidden_size",  # 隐藏维
        "intermediate_size",  # MLP 中间维
        "num_hidden_layers",  # 目标文本塔层数：界定 target_layer_ids 范围
        "num_attention_heads",  # 查询头数
        "num_global_key_value_heads",  # 全局 KV 头数（attention_k_eq_v 时作为 KV 头数）
        "global_head_dim",  # 全局头维：draft 注意力的 head_dim
        "attention_bias",  # QKVO 投影是否带偏置
        "attention_dropout",  # 注意力 dropout
        "attention_k_eq_v",  # 是否启用 K≡V 注意力变体（省略 v_proj，v 直接取 k）
        "enable_moe_block",  # 是否启用 MoE 块（draft 原型不支持，须为 False）
        "head_dim",  # 局部头维
        "hidden_activation",  # MLP 激活函数
        "hidden_size_per_layer_input",  # Gemma4 per-layer input 门控维（draft 原型需为 0）
        "initializer_range",  # 权重初始化标准差
        "max_position_embeddings",  # 最大位置数（RoPE 表长度）
        "num_key_value_heads",  # 局部 KV 头数
        "num_kv_shared_layers",  # KV 跨层共享的分组间隔
        "rms_norm_eps",  # RMSNorm 数值稳定项
        "rope_parameters",  # RoPE 参数（含 scaling 配置）
        "use_double_wide_mlp",  # 是否使用加宽 MLP
    )
    for field in required_fields:
        assert hasattr(text_config, field), (
            f"target_config.text_config.{field} must be provided."
        )  # 逐项检查，报错信息指明缺失字段名


def build_draft_config(target_config, model_args):
    # 由目标配置派生 DSpark draft 配置：与 Qwen3 版同构，差异仅在于先解包 Gemma4 多模态配置并校验字段。
    draft_config = get_gemma4_text_config(target_config)  # 取出目标文本配置的深拷贝
    _validate_required_text_fields(draft_config)  # 确保后续建模所需的 Gemma4 专有字段齐全

    num_target_layers = int(draft_config.num_hidden_layers)  # 目标文本塔层数（此刻 draft_config 仍是目标配置副本）
    num_draft_layers = int(model_args.num_draft_layers)  # draft 自身层数（通常 1-2 层）
    layer_types = ["full_attention"] * num_draft_layers  # draft 每层均为全注意力：块内自注意 + 跨注意，不用滑窗

    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."  # 必须显式指定从目标哪几层取特征
    target_layer_ids = validate_target_layer_ids(  # 校验并规范化为 int 列表
        model_args.target_layer_ids,  # 如 [2, 6, 10, 14, 18]，或含 -1（embedding 层输出）
        num_target_layers,  # 用于取值范围检查
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)  # 置信度损失权重 α：0 表示关闭置信度头
    assert confidence_head_alpha >= 0.0  # 权重不允许为负
    enable_confidence_head = confidence_head_alpha > 0.0  # α>0 才构建 AcceptRatePredictor（供推理期置信度调度）
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )  # 开置信度头时必须显式声明输入是否拼接 Markov 低秩隐变量

    markov_rank = int(model_args.markov_rank)  # Markov 头低秩转移矩阵的秩 r；0 表示不使用修正头
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )  # 秩>0 时必须指定头类型：vanilla / gated / rnn

    draft_config.architectures = ["Gemma4DSparkModel"]  # HF 按该架构名反射加载 Gemma4DSparkModel
    draft_config.target_model_type = str(target_config.model_type)  # 记录目标顶层类型（gemma4/gemma4_unified），供加载校验
    draft_config.target_text_model_type = str(draft_config.model_type)  # 记录目标文本子类型（gemma4_text 等）
    draft_config.num_target_layers = num_target_layers  # 记录目标层数，供特征抽取与校验使用
    draft_config.num_hidden_layers = num_draft_layers  # 核心改写：draft 实际层数替换目标层数
    draft_config.block_size = int(model_args.block_size)  # 每个 anchor 块内一次并行预测的 token 数（半自回归块长度）
    draft_config.tie_word_embeddings = False  # draft 的 lm_head 与 embed_tokens 不共享权重（两者均从目标拷贝）
    draft_config.layer_types = layer_types  # 覆盖为 draft 的全注意力层类型列表
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION  # 训练走 flex_attention 块稀疏路径
    draft_config.mask_token_id = int(model_args.mask_token_id)  # draft 块内噪声占位 token 的词表 id（块首会被真实 anchor token 覆盖）
    draft_config.target_layer_ids = target_layer_ids  # 从目标模型取特征的层 id 列表
    draft_config.num_anchors = int(model_args.num_anchors)  # 每条序列采样的 anchor 块数
    draft_config.enable_confidence_head = enable_confidence_head  # 是否构建置信度头
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )  # 置信度头输入是否拼接 Markov 低秩前驱嵌入
    draft_config.markov_rank = markov_rank  # Markov 头秩
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)  # Markov 头类型字符串
    return draft_config  # 返回可直接用于 from_config/from_pretrained 的 draft 配置


__all__ = [
    "build_draft_config",  # 对外导出配置工厂
    "get_gemma4_text_config",  # 供外部复用"顶层配置 → 文本配置"的解包逻辑
]
