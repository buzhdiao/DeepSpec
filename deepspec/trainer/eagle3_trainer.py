from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # 加载 target 模型配置与词表

from deepspec.data import CacheCollator  # 目标缓存样本整理器
from deepspec.modeling.eagle3.gemma4 import Gemma4Eagle3Model  # Gemma4 家族的 EAGLE3 draft 模型
from deepspec.modeling.eagle3.gemma4.config import (
    build_draft_config as build_gemma4_eagle3_config,  # 由 target 配置推导 Gemma4 EAGLE3 draft 配置
)
from deepspec.modeling.eagle3.loss import compute_eagle3_loss  # EAGLE3 多步（TTT 展开训练）损失
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model  # Qwen3 家族的 EAGLE3 draft 模型
from deepspec.modeling.eagle3.qwen3.config import (
    build_draft_config as build_qwen3_eagle3_config,  # 由 target 配置推导 Qwen3 EAGLE3 draft 配置
)
from deepspec.trainer.base_trainer import BaseTrainer  # 通用训练基类


class Qwen3Eagle3Trainer(BaseTrainer):
    # EAGLE3 算法的 Qwen3 训练器：与 BaseTrainer 的差异点——
    # 1) build_models 不用基类版本（冻结策略注释不同，流程等价）；
    # 2) run_batch 走 EAGLE3 特有的 TTT 展开式损失（跨多个 draft 步累计监督）。
    data_collator_cls = CacheCollator  # 缓存样本批整理器

    def build_models(self):
        # 重写模型构建：加载 tokenizer/target 配置 → 建 draft → 从 target 继承冻结嵌入与 lm_head。
        # 整体流程与基类一致，区别仅在于这里显式说明头部与 norm 的冻结策略（见下注释）。
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(  # 词表/chat template 取自 target 模型
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(  # target 结构信息
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(  # 构建 EAGLE3 draft（含三流输入融合等结构）
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        target_model = AutoModelForCausalLM.from_pretrained(  # CPU 加载 target 仅为继承嵌入/头
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()  # 输入嵌入矩阵
        target_lm_head = target_model.get_output_embeddings()  # 输出投影
        assert (target_lm_head is not None) and (target_embed_tokens is not None)

        # The draft head and norm stay frozen / target-independent to match
        # the DSpark setup: head is not trained and norm is not inherited.
        # 继承并冻结嵌入与 lm_head（与 DSpark 设置保持一致）：
        # draft 头不参与训练、norm 不从 target 继承，保证两算法的 draft 训练口径统一。
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )

        del target_model  # 立即释放 target 内存
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        # 按 target 配置推导 EAGLE3 draft 配置并实例化 Qwen3 模型。
        draft_config = build_qwen3_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3Eagle3Model(draft_config)

    def run_batch(self, batch):
        # 单 micro-batch 训练步：EAGLE3 的 TTT（test-time training 展开式）损失——
        # 在 draft 上沿时间维展开多步投机草稿，各步的 token/隐状态预测
        # 用目标缓存里的多层监督信号打分，按 step_loss_decay 加权求和。
        return compute_eagle3_loss(
            model=self.model,  # FSDP 封装后的训练模型
            batch=batch,  # 目标缓存批（input_ids/loss_mask/多层隐状态）
            ttt_length=int(self.draft_model.ttt_length),  # TTT 展开步数（草稿长度）
            step_loss_decay=float(self.draft_model.step_loss_decay),  # 各展开步的损失衰减系数
        )


class Gemma4Eagle3Trainer(Qwen3Eagle3Trainer):
    # EAGLE3 算法的 Gemma4 训练器：训练循环与损失复用 Qwen3 版实现，仅替换模型构建。
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4Eagle3Model(draft_config)
