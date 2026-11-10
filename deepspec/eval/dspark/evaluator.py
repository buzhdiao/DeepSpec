from __future__ import annotations  # 延迟注解求值：允许注解中使用 X | None 等新语法

from pathlib import Path  # 路径对象：artifact 目录拼接
from types import SimpleNamespace  # 属性式命名空间：draft 侧上下文状态

import torch  # PyTorch：张量操作
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # HF：自动加载 target 模型/tokenizer 与 KV cache

from deepspec.eval.base_evaluator import (  # 基础评测框架
    BaseEvaluator,  # 评测器基类
    DraftProposal,  # 通用提案数据类
    VerificationResult,  # 验证结果数据类
    assert_no_final_target_layer,  # 校验 draft 抽取层不含 target 最后一层
    generate_decoding_sample,  # 投机解码生成主循环
)
from deepspec.eval.dspark.confidence_head import ConfidenceHeadRecorder  # 置信度头校准统计记录器
from deepspec.eval.dspark.draft_ops import (  # dSpark 提案工具
    DSparkDraftProposal,  # 带置信度 logits 的提案
    build_dspark_proposal,  # 采样整块 token 并按置信度截断，构造提案
    forward_dspark_draft_block,  # draft 块一次非自回归前向
)
from deepspec.modeling.dspark.common import extract_context_feature  # 从 target 隐状态中抽取 draft 所需中间层特征
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel  # Gemma4 版 dSpark draft 模型
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # Qwen3 版 dSpark draft 模型
from deepspec.utils import jsonable  # 转可 JSON 序列化结构


CONFIDENCE_NUM_BINS = 20  # 置信度 ECE 粗分箱数
CONFIDENCE_NUM_FINE_BINS = 1000  # 置信度 AUROC 细分箱数


# Qwen3 目标模型的 dSpark 评测器：draft 一次非自回归前向并行产出整块 token
# （多 token 预测），可选置信度头按阈值截断提案长度；并把置信度校准指标
# （ECE/AUROC/Brier/可靠性图）接入评测上报链路。
class Qwen3DSparkEvaluator(BaseEvaluator):
    EVAL_ATTN_IMPLEMENTATION = "sdpa"  # 评测统一用 sdpa 注意力实现
    draft_model_cls = Qwen3DSparkModel  # draft 模型类；Gemma4 子类覆写为 Gemma4 版

    # 基类初始化（建模型）后按需构建置信度记录器
    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        self.confidence_head_recorder = self._build_confidence_head_recorder()

    # 每轮提案上限 = draft 的块大小（一次前向产出的 token 数）
    @property
    def max_proposal_tokens(self) -> int:
        return int(self.draft_model.block_size)

    # 构建置信度记录器（可选）：draft 无置信度头或配置了非零截断阈值时不启用——
    # 阈值截断会让提案长度随置信度波动，逐位置统计失去完整覆盖，校准无意义。
    def _build_confidence_head_recorder(self) -> ConfidenceHeadRecorder | None:
        if self.draft_model.confidence_head is None:
            return None
        if float(self.args.confidence_threshold) != 0.0:
            return None

        artifact_root = None  # artifact 目录跟随 TensorBoard 目录
        if self.args.tensorboard_dir is not None:
            artifact_root = (
                Path(self.args.tensorboard_dir)
                / "artifacts"
                / f"step_{self.args.step}"  # 按训练步数分目录，避免多次评测互相覆盖
            )
        return ConfidenceHeadRecorder(
            device=self.device,
            max_proposal_tokens=self.max_proposal_tokens,
            num_bins=CONFIDENCE_NUM_BINS,
            num_fine_bins=CONFIDENCE_NUM_FINE_BINS,
            draft_name_or_path=self.args.draft_name_or_path,
            tensorboard_dir=self.args.tensorboard_dir,
            step=self.args.step,
            artifact_root=artifact_root,
        )

    # 加载 target（HF AutoModel）与 dSpark draft 模型（bf16 + sdpa、eval 模式），并做一致性校验
    def build_models(self) -> tuple[object, Qwen3DSparkModel, AutoTokenizer]:
        target_model = AutoModelForCausalLM.from_pretrained(  # target 模型：标准 HF CausalLM
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()

        draft_model = self.draft_model_cls.from_pretrained(  # draft 模型：dSpark 自定义结构
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)  # 抽取层不能是 target 最后一层
        assert 0.0 <= float(self.args.confidence_threshold) <= 1.0  # 置信度阈值必须落在 [0,1]
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)  # tokenizer 与 target 对齐
        return target_model, draft_model, tokenizer

    # prefill 后构建 draft 侧上下文：空的 draft KV cache + 从 target 中间层抽取的
    # prompt 隐状态特征（dSpark 以 target 隐状态为条件输入）
    def _init_context(
        self,
        *,
        initial_output,  # target prefill 输出（含 hidden_states）
        **kwargs,  # 兼容钩子签名的其余参数（本实现未用）
    ) -> SimpleNamespace:
        return SimpleNamespace(
            past_key_values_draft=DynamicCache(),  # draft KV cache 从零开始
            target_hidden_states=extract_context_feature(  # 抽取 target 指定层的隐状态特征（支持 -1 表示 embedding 层）
                initial_output.hidden_states,
                self.draft_model.target_layer_ids,
            ),
        )

    # 一轮提案：把 [当前 token + mask token 块] 喂给 draft backbone（条件于 target
    # 隐状态），一次前向产出整块隐状态，再采样整块 token 并按置信度阈值截断成提案
    def _propose(
        self,
        *,
        context: SimpleNamespace,  # draft 侧上下文（KV cache + target 特征）
        output_ids: torch.Tensor,  # 当前完整序列
        position_ids: torch.Tensor,  # 全局位置 id
        start: int,  # 当前生成起始位置
        stop_token_ids: list[int] | None = None,  # 未用（dSpark 的提前截断走置信度阈值）
    ) -> DraftProposal:
        model = self.draft_model
        draft_input_ids = torch.full(  # draft 输入：整块长度，除首列外全部填 mask token
            (output_ids.size(0), self.max_proposal_tokens),
            int(model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]  # 首列为当前已接受 token
        block_hidden = forward_dspark_draft_block(  # 一次前向并行算整块隐状态
            model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states,
            start=start,
            block_size=self.max_proposal_tokens,
        )
        return build_dspark_proposal(  # 采样整块 token + 置信度截断，构造提案
            model=model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            confidence_threshold=float(self.args.confidence_threshold),
        )

    # 验证通过后推进 draft 侧状态：从本轮 target 验证输出重新抽取隐状态特征，
    # 截到"接受的 draft token + bonus token"长度——被拒位置的特征作废，
    # 下一轮提案条件于已验证的上下文
    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        verified_target_hidden = extract_context_feature(  # 本轮验证段各位置的 target 特征
            verification.target_output.hidden_states,
            self.draft_model.target_layer_ids,
        )
        context.target_hidden_states = verified_target_hidden[
            :,
            : verification.accepted_draft_tokens + 1,  # 只保留已提交部分（含 bonus token）
            :,
        ]

    # post_verify 钩子：把本轮提案与验证结果喂给置信度记录器做校准统计
    def _post_verify(
        self,
        proposal: DraftProposal,
        verification: VerificationResult,
    ) -> None:
        if self.confidence_head_recorder is None:  # 未启用记录器
            return
        assert isinstance(proposal, DSparkDraftProposal)  # 需要携带置信度 logits
        self.confidence_head_recorder.observe(
            proposal=proposal,
            verification=verification,
        )

    # 单样本生成：把各钩子注入通用投机解码循环
    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,  # 编码后的 prompt [1, L]
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
            post_verify=self._post_verify,  # 置信度校准统计入口
        )

    # 评测主流程：在基类基础上为每个数据集增加置信度记录的生命周期
    # （start -> 生成期 observe -> finish 归并 -> report 打印/落盘）
    def evaluate(self) -> None:
        for dataset_name, max_samples in self.tasks:
            if self.confidence_head_recorder is not None:
                self.confidence_head_recorder.start()  # 重建流式统计器
            responses = self.run_dataset(  # 生成各 rank 分片的样本
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            metric_summary = self.allreduce_response_metrics(responses)  # 跨 rank 汇总投机解码指标
            confidence_row = (
                self.confidence_head_recorder.finish(  # 归并置信度统计并产出指标行
                    dataset_name=dataset_name,
                    metric_summary=metric_summary,
                )
                if self.confidence_head_recorder is not None
                else None
            )

            metrics_row = self.record_dataset_metrics(  # rank 0 记录并打印投机解码指标
                dataset_name=dataset_name,
                metric_summary=metric_summary,
            )
            if metrics_row is not None and confidence_row is not None:
                self.confidence_head_recorder.report_dataset(  # 打印/落盘置信度结果
                    metrics_row=metrics_row,
                    confidence_row=confidence_row,
                    args_payload=jsonable(vars(self.args)),  # 运行参数快照写入 artifact
                    tasks=self.tasks,
                )

        self.report_results()

    # 追加置信度指标的 TensorBoard 上报
    def log_tensorboard(self) -> None:
        super().log_tensorboard()
        if self.confidence_head_recorder is not None:
            self.confidence_head_recorder.log_tensorboard()

    # 追加置信度汇总表打印
    def print_results(self) -> None:
        super().print_results()
        if self.confidence_head_recorder is not None:
            self.confidence_head_recorder.print_results()


# Gemma4 目标模型的 dSpark 评测器：仅替换 draft 模型类，其余逻辑与 Qwen3 版完全一致
class Gemma4DSparkEvaluator(Qwen3DSparkEvaluator):
    draft_model_cls = Gemma4DSparkModel
