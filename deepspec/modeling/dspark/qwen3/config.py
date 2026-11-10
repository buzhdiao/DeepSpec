# Qwen3 版 DSpark draft 配置构造：深拷贝目标配置派生 draft 配置，改写层数并写入 DSpark 专有字段。
import copy  # 深拷贝目标配置，draft 与 target 各持一份配置对象，避免共享可变状态

from deepspec.modeling.dspark.common import validate_target_layer_ids  # 校验 target_layer_ids：非空、严格递增、范围合法（-1 表示 embedding 输出）


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"  # 训练期注意力实现固定为 flex_attention，以使用 create_block_mask 的块稀疏掩码


def build_draft_config(
    target_config,  # 目标模型（Qwen3）配置：提供 vocab_size/hidden_size/RoPE 等需与目标对齐的基础超参
    model_args,  # draft 训练超参容器：num_draft_layers/block_size/anchor 抽样/Markov 头/置信度头等字段
):
    # 由目标配置派生 DSpark draft 配置：deepcopy 保留目标全部生成相关超参（eos/bos/max_position 等），
    # 仅把层数换成 num_draft_layers，再写入 anchor 抽样、Markov 头、置信度头等 DSpark 专有字段。
    num_target_layers = int(target_config.num_hidden_layers)  # 目标模型层数：界定 target_layer_ids 的合法范围
    num_draft_layers = int(model_args.num_draft_layers)  # draft 自身层数（通常 1-2 层，远小于目标层数）
    layer_types = ["full_attention"] * num_draft_layers  # draft 每层均为全注意力：块内自注意 + 对目标上下文的跨注意，不用滑窗
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."  # 必须显式指定从目标哪几层取特征（多层级特征融合是 DSpark/Eagle3 的关键）
    target_layer_ids = validate_target_layer_ids(  # 校验并规范化为 int 列表
        model_args.target_layer_ids,  # 如 [2, 6, 10, 14, 18]，或含 -1（embedding 层输出）
        num_target_layers,  # 用于取值范围检查
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)  # 置信度损失权重 α：0 表示关闭置信度头
    assert confidence_head_alpha >= 0.0  # 权重不允许为负
    enable_confidence_head = confidence_head_alpha > 0.0  # α>0 才构建 AcceptRatePredictor（预测逐位置接受率，供推理期置信度调度）
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

    draft_config = copy.deepcopy(target_config)  # 深拷贝：draft 继承目标的一切词表/位置/采样超参
    draft_config.architectures = ["Qwen3DSparkModel"]  # HF 按该架构名反射加载 Qwen3DSparkModel
    draft_config.num_target_layers = num_target_layers  # 记录目标层数，供特征抽取与校验使用
    draft_config.num_hidden_layers = num_draft_layers  # 核心改写：draft 实际层数替换目标层数
    draft_config.block_size = int(model_args.block_size)  # 每个 anchor 块内 draft 一次并行预测的 token 数（半自回归块长度）
    draft_config.tie_word_embeddings = False  # draft 的 lm_head 与 embed_tokens 不共享权重（两者均从目标拷贝，可独立冻结/训练）
    draft_config.layer_types = layer_types  # 覆盖为 draft 的全注意力层类型列表
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION  # 训练走 flex_attention 块稀疏路径
    draft_config.mask_token_id = int(model_args.mask_token_id)  # draft 块内噪声占位 token 的词表 id（块首会被真实 anchor token 覆盖）
    draft_config.target_layer_ids = target_layer_ids  # 从目标模型取特征的层 id 列表
    draft_config.num_anchors = int(model_args.num_anchors)  # 每条序列采样的 anchor 块数
    draft_config.enable_confidence_head = enable_confidence_head  # 是否构建置信度头
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )  # 置信度头输入是否拼接 Markov 低秩前驱嵌入（增强对上一 token 的感知）
    draft_config.markov_rank = markov_rank  # Markov 头秩
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)  # Markov 头类型字符串
    return draft_config  # 返回可直接用于 from_config/from_pretrained 的 draft 配置


__all__ = [
    "build_draft_config",  # 对外只导出配置工厂
]
