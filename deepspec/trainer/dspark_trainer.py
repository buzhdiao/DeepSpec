from deepspec.data import CacheCollator  # 目标缓存样本整理器：补零成批 + 重建 attention_mask
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel  # Gemma4 家族的 DSpark draft 模型
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,  # 由 target 配置推导 Gemma4 draft 配置
)
from deepspec.modeling.dspark.loss import compute_dspark_loss  # DSpark 复合损失（对照 CE/L1/置信度）
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # Qwen3 家族的 DSpark draft 模型
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,  # 由 target 配置推导 Qwen3 draft 配置
)
from deepspec.trainer.base_trainer import BaseTrainer  # 通用训练基类


class Qwen3DSparkTrainer(BaseTrainer):
    # DSpark 算法的 Qwen3 训练器：训练数据为目标缓存（离线物化的 target 隐状态监督信号），
    # run_batch 只做 draft 前向 + DSpark 复合损失，Gemma4 版仅替换模型构建部分。
    data_collator_cls = CacheCollator  # 缓存样本批整理：input_ids/loss_mask/隐状态分别补零

    def _build_draft_model(self, *, target_config, model_args):
        # 按 target 模型结构推导 draft 配置（层数/维度/词表）并实例化 DSpark 模型。
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        # 单 micro-batch 训练步：draft 前向吃 input_ids 与 target 多层/末层隐状态，
        # 输出各层预测与置信度，再用 DSpark 复合损失对齐监督信号。
        # loss_mask 决定哪些 token 计损失（仅 assistant 回复区）。
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],  # [B, L, L*H] 多中间层监督
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],  # [B, L, H] 末层监督
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,  # 多步预测损失的衰减系数
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),  # 对照 token CE 损失权重
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),  # 隐状态回归 L1 损失权重
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),  # 置信度头损失权重
        )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    # DSpark 算法的 Gemma4 训练器：训练循环/损失/数据管线与 Qwen3 版完全一致，
    # 仅把 draft 模型构建换成 Gemma4 家族（含 Gemma 特有的 thought 通道格式处理）。
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)
