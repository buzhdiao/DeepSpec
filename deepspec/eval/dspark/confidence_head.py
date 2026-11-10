from __future__ import annotations  # 延迟注解求值：允许注解中使用 X | None 等新语法

import json  # 标准库：JSON 序列化/反序列化
import math  # 标准库：isnan/isfinite 判断
import os  # 标准库：从模型路径取 basename 作显示名
from pathlib import Path  # 路径对象：artifact 目录与文件路径
from typing import Protocol  # 结构化类型：定义提案的鸭子类型协议

import torch  # PyTorch：张量统计（scatter_add、cumprod、cumsum 等）
import torch.distributed as dist  # torch 分布式：跨 rank 归并统计直方图

from deepspec.eval.base_evaluator import VerificationResult  # target 验证结果数据类
from deepspec.utils import jsonable  # 把任意对象转成可 JSON 序列化结构


EPS_PROB = 1e-8  # 概率截断下限：把预测夹进 (0,1)，避免 log/除法数值问题
RELIABILITY_PLOT_FILENAME = "reliability_diagram.png"  # 可靠性图的固定输出文件名


# 提案协议：ConfidenceHeadRecorder 只依赖这两个属性（鸭子类型），便于解耦与测试
class ConfidenceProposal(Protocol):
    draft_token_count: int  # 提案的 draft token 数
    confidence_logits: torch.Tensor | None  # 置信度头逐位置 logits


# 模型显示名：取路径 basename；根路径等 basename 为空的退化情况回退为原路径
def model_display_name(path: str) -> str:
    normalized = path.rstrip("/")
    return os.path.basename(normalized) or normalized


# 逐位置置信度校准指标累加器（ECE + AUROC + Brier）。
# 设计：样本流式到达（每次验证 ≤ block_size 个前缀位置），无法缓存全部样本，
# 因此按"位置 x 分箱"维护计数直方图，跑完后一次性出指标：
#   - 粗分箱直方图（count/pred/target 三份）供 ECE 与可靠性图；
#   - 细分箱正/负样本直方图供 AUROC 的 Mann-Whitney 近似；
#   - Brier 分子逐位置直接累加。
# 全部用 float64 + scatter_add_，保证多轮浮点累加精度。
class PerPositionConfidenceMetrics:
    """Per-position ECE + AUROC + Brier for cumprod predictions."""

    # 初始化各统计直方图（全零、float64、在指定设备上）
    def __init__(
        self,
        *,
        block_size: int,  # 位置数（= max_proposal_tokens）
        num_coarse_bins: int,  # ECE/可靠性图的粗分箱数
        num_fine_bins: int,  # AUROC 的细分箱数（直方图近似 ROC）
        device: torch.device,  # 统计张量所在设备
    ):
        self.block_size = int(block_size)
        self.num_coarse_bins = int(num_coarse_bins)
        self.num_fine_bins = int(num_fine_bins)
        self.coarse_count = torch.zeros(  # 粗分箱样本计数 [block_size, num_coarse_bins]
            (self.block_size, self.num_coarse_bins),
            dtype=torch.float64,
            device=device,
        )
        self.coarse_pred = torch.zeros_like(self.coarse_count)  # 粗分箱预测概率累加（除以计数得箱内均值预测）
        self.coarse_target = torch.zeros_like(self.coarse_count)  # 粗分箱真实标签累加（除以计数得箱内实际接受率）
        self.fine_pos = torch.zeros(  # 细分箱正样本（前缀被接受）计数，AUROC 用
            (self.block_size, self.num_fine_bins),
            dtype=torch.float64,
            device=device,
        )
        self.fine_neg = torch.zeros_like(self.fine_pos)  # 细分箱负样本（前缀被拒绝）计数
        self.brier_num = torch.zeros(  # Brier 分子逐位置累加：sum((pred − target)^2)
            self.block_size,
            dtype=torch.float64,
            device=device,
        )

    # 流式累加一批样本（一次验证的前缀位置序列）：
    #   - probs：cumprod 置信预测——"前缀到位置 i 全被接受"的预估概率；
    #   - targets：真实前缀接受标签（0/1，来自拒绝采样的前缀掩码）。
    # 把每个样本按预测值落入 (位置, 分箱) 格子散射累加：粗箱供 ECE，细箱供 AUROC，
    # 另逐位置累加 Brier 分子。
    def update(
        self,
        *,
        probs: torch.Tensor,  # [k] 前缀接受概率预测（cumprod）
        targets: torch.Tensor,  # [k] 前缀接受真实标签（0/1）
    ) -> None:
        probs = probs.reshape(-1).to(torch.float64).clamp(EPS_PROB, 1.0 - EPS_PROB)  # 展平、升精度、夹进开区间
        targets = targets.reshape(-1).to(torch.float64)  # 展平为 0/1 浮点
        assert probs.shape == targets.shape
        pos_count = probs.numel()  # 本批位置数
        assert pos_count <= self.block_size  # 不得超过统计器容量
        weights = torch.ones_like(probs, dtype=torch.float64)  # 全 1 权重：scatter_add_ 的计数载体
        pos_idx = torch.arange(pos_count, device=probs.device)  # 位置下标

        coarse_idx = (  # 粗分箱：prob * bins 取整并夹进合法区间
            probs * self.num_coarse_bins
        ).long().clamp_(0, self.num_coarse_bins - 1)
        flat_coarse = pos_idx * self.num_coarse_bins + coarse_idx  # 展平为一维的 (位置, 箱) 索引
        self.coarse_count.view(-1).scatter_add_(0, flat_coarse, weights)  # 箱计数 +1
        self.coarse_pred.view(-1).scatter_add_(0, flat_coarse, probs * weights)  # 箱内预测累加
        self.coarse_target.view(-1).scatter_add_(0, flat_coarse, targets * weights)  # 箱内标签累加

        fine_idx = (  # 细分箱：同规则、更细粒度（AUROC 分辨率更高）
            probs * self.num_fine_bins
        ).long().clamp_(0, self.num_fine_bins - 1)
        flat_fine = pos_idx * self.num_fine_bins + fine_idx
        self.fine_pos.view(-1).scatter_add_(0, flat_fine, weights * targets)  # 箱内正样本计数
        self.fine_neg.view(-1).scatter_add_(0, flat_fine, weights * (1.0 - targets))  # 箱内负样本计数

        self.brier_num[:pos_count].add_(weights * (probs - targets).pow(2))  # 逐位置累加 (pred − target)^2

    # 跨 rank 求和全部直方图：各 rank 只见自己分片的样本，归并后才是全局统计
    def all_reduce(self) -> None:
        for tensor in (
            self.coarse_count,
            self.coarse_pred,
            self.coarse_target,
            self.fine_pos,
            self.fine_neg,
            self.brier_num,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 由正/负样本分箱直方图计算 AUROC（Mann-Whitney U 统计量的直方图近似）：
    # 对每个正样本，累加"分数严格更低"的负样本数 + 0.5 * 同箱负样本数（并列计一半），
    # 再除以正负样本对总数归一化到 [0,1]。任一侧无样本时返回 NaN。
    @staticmethod
    def _auroc_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> float:
        total_pos = float(pos_hist.sum().item())  # 正样本总数
        total_neg = float(neg_hist.sum().item())  # 负样本总数
        if total_pos <= 0.0 or total_neg <= 0.0:
            return float("nan")
        cum_neg = torch.cumsum(neg_hist, dim=0)  # 负样本分数前缀和（含当前箱）
        cum_neg_before = cum_neg - neg_hist  # 严格低于当前箱的负样本数
        pair = (pos_hist * cum_neg_before).sum() + 0.5 * (pos_hist * neg_hist).sum()  # U 统计量（并列按 0.5 计）
        return float(pair.item() / (total_pos * total_neg))  # 归一化为 AUC

    # 由累加直方图计算每个位置的最终指标：
    #   ECE：箱内 |平均预测 − 实际接受率| 按样本数加权平均（期望校准误差）；
    #   AUC：细分箱直方图近似 AUROC（区分"前缀会被接受/拒绝"的判别力）；
    #   Brier：均方误差 (pred − target)^2 均值；
    # 另输出箱级可靠性数据（供画 reliability diagram）与全位置平均预测/接受率。
    # 无样本的位置返回 NaN 占位，reliability 为空列表。
    def compute(self) -> list[dict]:
        out = []
        for pos in range(self.block_size):
            weights = self.coarse_count[pos]  # 该位置各粗箱计数
            total = float(weights.sum().item())  # 该位置总样本数
            if total <= 1e-12:  # 无样本：NaN 占位
                out.append(
                    {
                        "position": pos,
                        "total_weight": 0.0,
                        "ece": float("nan"),
                        "auc": float("nan"),
                        "brier": float("nan"),
                        "pred_mean": float("nan"),
                        "target_mean": float("nan"),
                        "reliability": [],
                    }
                )
                continue

            denom = weights.clamp_min(1e-12)  # 防除零
            avg_pred = self.coarse_pred[pos] / denom  # 箱内平均预测
            avg_target = self.coarse_target[pos] / denom  # 箱内实际接受率
            bin_err = (avg_pred - avg_target).abs()  # 箱内校准误差
            ece = float((bin_err * weights).sum().item() / total)  # 按样本数加权平均
            auc = self._auroc_from_hist(self.fine_pos[pos], self.fine_neg[pos])  # 直方图 AUROC
            brier = float(self.brier_num[pos].item()) / total  # Brier 均值
            reliability = []  # 箱级可靠性数据（画图用）
            for bin_idx in range(self.num_coarse_bins):
                weight = float(weights[bin_idx].item())
                if weight <= 0.0:  # 空箱跳过
                    continue
                reliability.append(
                    {
                        "bin": bin_idx,
                        "range": [  # 箱的概率区间 [lo, hi)
                            bin_idx / self.num_coarse_bins,
                            (bin_idx + 1) / self.num_coarse_bins,
                        ],
                        "avg_pred": float(avg_pred[bin_idx].item()),
                        "avg_target": float(avg_target[bin_idx].item()),
                        "weight": weight,
                    }
                )
            out.append(
                {
                    "position": pos,
                    "total_weight": total,
                    "ece": ece,
                    "auc": auc,
                    "brier": brier,
                    "pred_mean": float(self.coarse_pred[pos].sum().item()) / total,  # 全位置平均预测
                    "target_mean": float(self.coarse_target[pos].sum().item()) / total,  # 全位置实际接受率
                    "reliability": reliability,
                }
            )
        return out


# 按位置样本数加权汇总整行指标：对各位置的 ece/brier/pred/target 做加权平均；
# AUC 只对非 NaN 位置加权平均（auc_w 为有效权重）。全行无样本时返回 NaN。
def summarize_confidence_row(row: dict) -> dict:
    total_w = 0.0  # 总权重（样本数）
    weighted_ece = 0.0
    weighted_brier = 0.0
    weighted_pred = 0.0
    weighted_target = 0.0
    auc_w = 0.0  # AUC 有效权重（排除 NaN 位置）
    weighted_auc = 0.0

    for entry in row.get("per_position") or []:
        weight = float(entry["total_weight"])
        if weight <= 0.0:  # 无样本位置跳过
            continue
        total_w += weight
        weighted_ece += float(entry["ece"]) * weight
        weighted_brier += float(entry["brier"]) * weight
        weighted_pred += float(entry["pred_mean"]) * weight
        weighted_target += float(entry["target_mean"]) * weight
        auc = float(entry["auc"])
        if not math.isnan(auc):  # NaN 的 AUC 不计入加权平均
            auc_w += weight
            weighted_auc += auc * weight

    if total_w <= 0.0:  # 全行无有效样本
        return {
            "ece_mean": float("nan"),
            "auc_mean": float("nan"),
            "brier_mean": float("nan"),
            "pred_mean": float("nan"),
            "target_mean": float("nan"),
        }
    return {
        "ece_mean": weighted_ece / total_w,
        "auc_mean": weighted_auc / auc_w if auc_w > 0.0 else float("nan"),  # 有效权重为 0 时仍为 NaN
        "brier_mean": weighted_brier / total_w,
        "pred_mean": weighted_pred / total_w,
        "target_mean": weighted_target / total_w,
    }


# 浮点格式化：NaN 显示为 "nan"，其余按指定位数定点输出（表格展示用）
def format_float(value, digits=4) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


# 压缩行数据用于终端打印/单行 JSON：去掉大体积的 reliability 明细并附加汇总指标；
# per_position 只保留标量字段（剔除 reliability 列表）。
def compact_row(row: dict) -> dict:
    out = {key: value for key, value in row.items() if key != "per_position"}
    out.update(summarize_confidence_row(row))  # 附加加权汇总指标
    out["per_position"] = [
        {key: value for key, value in entry.items() if key != "reliability"}
        for entry in row.get("per_position") or []
    ]
    return out


# 绘制逐位置可靠性图（reliability diagram）并保存为 PNG：
# 每个位置一个子图——橙色折线为"箱内平均预测 vs 实际接受率"（灰色虚线为理想对角线），
# 次坐标轴半透明柱状为各箱样本权重分布（看哪些箱有统计意义）；
# 缺位置用多余子图占位并关闭坐标轴。用 Agg 后端离屏渲染（服务器无显示器）。
def plot_reliability_diagram(dataset_dir: Path, row: dict) -> Path:
    import matplotlib  # 延迟导入画图库

    matplotlib.use("Agg")  # 无 GUI 后端：只写文件
    import matplotlib.pyplot as plt

    per_position = row["per_position"]
    num_positions = max(len(per_position), 1)  # 至少 1 个子图
    ncols = min(3, num_positions)  # 每行最多 3 列
    nrows = (num_positions + ncols - 1) // ncols  # 行数向上取整
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        squeeze=False,  # 保持二维数组索引方式
    )

    for idx, entry in enumerate(per_position):
        ax = axes[idx // ncols][idx % ncols]  # 行优先排布子图
        reliability = entry.get("reliability") or []
        ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.7", linewidth=1.0)  # 理想校准对角线
        if reliability:
            centers = [  # 各箱中心 x 坐标
                0.5 * (float(bin_row["range"][0]) + float(bin_row["range"][1]))
                for bin_row in reliability
            ]
            widths = [  # 各箱宽度
                float(bin_row["range"][1]) - float(bin_row["range"][0])
                for bin_row in reliability
            ]
            avg_pred = [float(bin_row["avg_pred"]) for bin_row in reliability]  # 箱内平均预测
            avg_target = [float(bin_row["avg_target"]) for bin_row in reliability]  # 箱内实际接受率
            weights = [float(bin_row["weight"]) for bin_row in reliability]  # 箱内样本权重
            ax.plot(avg_pred, avg_target, "o-", color="C1", linewidth=1.5, markersize=4)  # 校准曲线
            ax2 = ax.twinx()  # 次坐标轴画权重直方图
            ax2.bar(
                centers,
                weights,
                width=widths,
                color="C0",
                alpha=0.2,
                edgecolor="white",
                linewidth=0.5,
            )
            max_weight = max(weights) if weights else 0.0
            ax2.set_ylim(0.0, max_weight if max_weight > 0.0 else 1.0)  # 柱高以最大权重为基准
            ax2.set_yticks([])  # 不显示次轴刻度（权重只是辅助信息）
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted prefix acceptance")
        ax.set_ylabel("Observed prefix acceptance")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_title(  # 标题携带该位置核心指标
            "pos={pos}  ECE={ece:.4f}  AUC={auc:.4f}\nmean_pred={pred:.4f}  "
            "mean_target={target:.4f}".format(
                pos=int(entry["position"]),
                ece=float(entry["ece"]),
                auc=float(entry["auc"]),
                pred=float(entry["pred_mean"]),
                target=float(entry["target_mean"]),
            ),
            fontsize=10,
        )

    for idx in range(len(per_position), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")  # 多余子图占位后关闭

    fig.suptitle(f"{row['dataset']} confidence-head reliability", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))  # 给总标题留出顶部空间
    output_path = dataset_dir / RELIABILITY_PLOT_FILENAME
    fig.savefig(output_path, dpi=200)
    plt.close(fig)  # 释放画布内存
    return output_path


# 写 TensorBoard 标量的安全包装：NaN/Inf 指标跳过，避免污染曲线
def add_tensorboard_scalar(writer, tag: str, value, step: int) -> None:
    value = float(value)
    if math.isfinite(value):
        writer.add_scalar(tag, value, step)


# 置信度头评测记录器：挂接在投机解码循环的 post_verify 钩子上，流式收集
# "cumprod 置信预测 vs 实际前缀接受"样本；跑完一个数据集后跨 rank 归并统计，
# 产出逐位置指标行（ECE/AUROC/Brier）、可靠性图与 TensorBoard 曲线。
# 仅在 confidence_threshold=0（不按置信度截断提案）时启用——此时每轮提案都是
# 完整块长，逐位置统计才有完整覆盖。
class ConfidenceHeadRecorder:

    # 保存配置；dataset_metrics 在 start() 时创建，rows 收集各数据集的指标行
    def __init__(
        self,
        *,
        device: torch.device,  # 统计张量设备
        max_proposal_tokens: int,  # 位置数（= block_size）
        num_bins: int,  # ECE 粗分箱数
        num_fine_bins: int,  # AUROC 细分箱数
        draft_name_or_path: str,  # draft 模型路径（展示与落盘用）
        tensorboard_dir: str | None,  # TensorBoard 目录（None 则不写曲线）
        step: int | None,  # 训练步数（TensorBoard global_step）
        artifact_root: Path | None,  # artifact 输出根目录（None 则不落盘）
    ):
        self.device = device
        self.max_proposal_tokens = int(max_proposal_tokens)
        self.num_bins = int(num_bins)
        self.num_fine_bins = int(num_fine_bins)
        self.draft_name_or_path = draft_name_or_path
        self.tensorboard_dir = tensorboard_dir
        self.step = step
        self.artifact_root = artifact_root
        self.dataset_metrics: PerPositionConfidenceMetrics | None = None  # 当前数据集的流式统计器
        self.rows: list[dict] = []  # 各数据集的置信度指标行

    # 开始一个数据集：重建流式统计器（上一数据集的数据已在 finish 中归并并释放）
    def start(self) -> None:
        self.dataset_metrics = PerPositionConfidenceMetrics(
            block_size=self.max_proposal_tokens,
            num_coarse_bins=self.num_bins,
            num_fine_bins=self.num_fine_bins,
            device=self.device,
        )

    # post_verify 钩子：一次验证后收集校准样本。
    # 预测取逐位置置信概率的 cumprod——得到"前缀到位置 i 全被接受"的联合预估；
    # 标签取拒绝采样的 accept_prefix_mask（本身也是 cumprod 掩码），两者逐位置对齐。
    # 截断到 effective_proposal_length：stop token 之后的提案位置不参与统计。
    def observe(
        self,
        *,
        proposal: ConfidenceProposal,  # dSpark 提案（须携带置信度 logits）
        verification: VerificationResult,  # target 验证结果
    ) -> None:
        assert self.dataset_metrics is not None  # 必须先 start()
        if int(proposal.draft_token_count) <= 0:  # 空提案无统计意义
            return
        effective_length = int(verification.effective_proposal_length)
        if effective_length <= 0:  # 生效长度为 0：无样本
            return
        confidence_logits = proposal.confidence_logits
        assert confidence_logits is not None
        assert verification.accept_prefix_mask is not None
        # BaseEvaluator.generate_one_sample enforces bsz=1, so this removes
        # only the single-sequence batch dimension. Truncate to
        # effective_proposal_length to skip positions past an accepted EOS.
        step_probs = torch.sigmoid(  # logits -> 逐位置"该步被接受"概率，并截到有效长度
            confidence_logits[:, :effective_length]
        ).squeeze(0)  # 去掉 bsz=1 的批维
        cumprod_pred = step_probs.to(torch.float64).cumprod(dim=0)  # 连乘得前缀接受概率预估
        prefix_label = (  # 真实前缀接受标签（拒绝采样的前缀掩码）
            verification.accept_prefix_mask[:, :effective_length]
            .squeeze(0)
            .to(torch.float64)
        )
        self.dataset_metrics.update(
            probs=cumprod_pred,
            targets=prefix_label,
        )

    # 结束一个数据集：跨 rank 归并统计后释放；rank 0 且有样本时构建指标行并缓存。
    # 返回指标行（供 report_dataset 打印/落盘）；非 rank 0 返回 None。
    def finish(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | list[int]],  # 投机解码侧汇总计数（判断样本数）
    ) -> dict | None:
        assert self.dataset_metrics is not None
        dataset_metrics = self.dataset_metrics
        dataset_metrics.all_reduce()  # 跨 rank 求和全部直方图
        self.dataset_metrics = None  # 释放：下一数据集需重新 start()

        if dist.get_rank() != 0 or int(metric_summary["sample_count"]) == 0:  # 非 rank 0 或无样本
            return None

        row = self.build_dataset_row(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
            confidence_metrics=dataset_metrics,
        )
        self.rows.append(row)  # 缓存供汇总表格/TensorBoard
        return row

    # 构建单数据集置信度指标行：元信息 + 逐位置指标（compute 的输出）
    def build_dataset_row(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | list[int]],
        confidence_metrics: PerPositionConfidenceMetrics,
    ) -> dict:
        return {
            "dataset": dataset_name,
            "sample_count": int(metric_summary["sample_count"]),  # 样本数
            "proposal_count": int(metric_summary["proposal_count"]),  # 验证轮数
            "draft_name_or_path": self.draft_name_or_path,
            "draft_name": model_display_name(self.draft_name_or_path),
            "per_position": confidence_metrics.compute(),  # 逐位置 ECE/AUC/Brier/可靠性数据
        }

    # 上报单数据集置信度结果：rank 0 打印紧凑 JSON（键排序保证输出稳定）；
    # 配置了 artifact 目录时把 metrics.json 与可靠性图落盘。
    def report_dataset(
        self,
        *,
        metrics_row: dict[str, object],  # 投机解码侧指标行（写入 metrics.json 的 spec 字段）
        confidence_row: dict,  # 置信度指标行
        args_payload: dict,  # 序列化后的运行参数快照
        tasks: list[tuple[str, int | None]],  # 任务列表（写入 config）
    ) -> None:
        print(
            json.dumps(
                compact_row(confidence_row),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if self.artifact_root is not None:
            metrics_path, extra_paths = self.write_dataset_outputs(
                metrics_row=metrics_row,
                confidence_row=confidence_row,
                args_payload=args_payload,
                tasks=tasks,
            )
            print(f"Wrote dataset metrics to {metrics_path}", flush=True)
            for path in extra_paths:
                print(f"Wrote dataset artifact to {path}", flush=True)

    # 组装 metrics.json 的完整内容：config（参数 + 任务）+ 投机解码指标 + 置信度指标与加权汇总
    def build_dataset_payload(
        self,
        *,
        metrics_row: dict[str, object],
        confidence_row: dict,
        args_payload: dict,
        tasks: list[tuple[str, int | None]],
    ) -> dict:
        return {
            "config": {
                "args": args_payload,
                "tasks": jsonable(tasks),  # 转成可 JSON 序列化的结构
            },
            "spec": metrics_row,  # 投机解码侧指标
            "confidence": confidence_row,  # 置信度逐位置指标
            "confidence_summary": summarize_confidence_row(confidence_row),  # 加权汇总
        }

    # 落盘单数据集产物：写 metrics.json 并绘制可靠性图，返回文件路径供打印
    def write_dataset_outputs(
        self,
        *,
        metrics_row: dict[str, object],
        confidence_row: dict,
        args_payload: dict,
        tasks: list[tuple[str, int | None]],
    ) -> tuple[Path, list[Path]]:
        assert self.artifact_root is not None
        dataset_dir = self.artifact_root / str(metrics_row["dataset"])  # 每个数据集独立子目录
        dataset_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = dataset_dir / "metrics.json"
        payload = self.build_dataset_payload(
            metrics_row=metrics_row,
            confidence_row=confidence_row,
            args_payload=args_payload,
            tasks=tasks,
        )
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)  # 缩进 2 便于人工检查
        extra_paths = [plot_reliability_diagram(dataset_dir, confidence_row)]  # 可靠性图
        return metrics_path, extra_paths

    # 把置信度指标写入 TensorBoard：逐位置的 ece/auc/brier/pred_mean/target_mean
    # 五组标量，外加每数据集的加权汇总指标；NaN 值由 add_tensorboard_scalar 过滤
    def log_tensorboard(self) -> None:
        if not self.rows:
            return

        from torch.utils.tensorboard import SummaryWriter  # 延迟导入

        assert self.tensorboard_dir is not None
        assert self.step is not None
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for row in self.rows:
            dataset = row["dataset"]
            per_position = row.get("per_position") or []
            for entry in per_position:
                pos = int(entry["position"])
                weight = float(entry["total_weight"])
                if weight <= 0.0:  # 无样本位置不写
                    continue
                add_tensorboard_scalar(  # 期望校准误差
                    writer,
                    f"confidence/{dataset}/ece@{pos}",
                    entry["ece"],
                    self.step,
                )
                add_tensorboard_scalar(  # 判别力 AUROC
                    writer,
                    f"confidence/{dataset}/auc@{pos}",
                    entry["auc"],
                    self.step,
                )
                add_tensorboard_scalar(  # Brier 分数
                    writer,
                    f"confidence/{dataset}/brier@{pos}",
                    entry["brier"],
                    self.step,
                )
                add_tensorboard_scalar(  # 平均预测
                    writer,
                    f"confidence/{dataset}/pred_mean@{pos}",
                    entry["pred_mean"],
                    self.step,
                )
                add_tensorboard_scalar(  # 实际前缀接受率
                    writer,
                    f"confidence/{dataset}/target_mean@{pos}",
                    entry["target_mean"],
                    self.step,
                )

            for key, value in summarize_confidence_row(row).items():
                add_tensorboard_scalar(  # 加权汇总指标（ece_mean/auc_mean/...）
                    writer,
                    f"confidence/{dataset}/{key}",
                    value,
                    self.step,
                )
        writer.close()

    # 渲染置信度汇总表：基础列 + 逐位置 ece/auc 列（列数取各数据集最大位置数）
    def build_table(self) -> str:
        from prettytable import PrettyTable  # 延迟导入

        table = PrettyTable()
        max_position_count = max(  # 逐位置列的最大数量
            (len(row.get("per_position") or []) for row in self.rows),
            default=0,
        )
        field_names = [  # 基础列：数据集/draft 模型/样本数/轮数 + 加权汇总指标
            "dataset",
            "draft_model",
            "samples",
            "proposals",
            "ece_mean",
            "auc_mean",
            "brier_mean",
            "pred_mean",
            "target_mean",
        ]
        field_names.extend(f"ece@{pos}" for pos in range(max_position_count))  # 逐位置 ECE 列
        field_names.extend(f"auc@{pos}" for pos in range(max_position_count))  # 逐位置 AUC 列
        table.field_names = field_names

        draft_name = model_display_name(self.draft_name_or_path)
        for row in self.rows:
            summary = summarize_confidence_row(row)  # 加权汇总指标
            per_position = {  # 位置下标 -> 指标项，便于按列取值
                int(entry["position"]): entry
                for entry in row.get("per_position") or []
            }
            table.add_row(
                [
                    row["dataset"],
                    draft_name,
                    row["sample_count"],
                    row["proposal_count"],
                    format_float(summary["ece_mean"]),
                    format_float(summary["auc_mean"]),
                    format_float(summary["brier_mean"]),
                    format_float(summary["pred_mean"]),
                    format_float(summary["target_mean"]),
                ] + [  # 逐位置 ECE：无样本位置显示 "-"
                    (
                        format_float(per_position[pos]["ece"])
                        if pos in per_position
                        and float(per_position[pos]["total_weight"]) > 0.0
                        else "-"
                    )
                    for pos in range(max_position_count)
                ] + [  # 逐位置 AUC：无样本位置显示 "-"
                    (
                        format_float(per_position[pos]["auc"])
                        if pos in per_position
                        and float(per_position[pos]["total_weight"]) > 0.0
                        else "-"
                    )
                    for pos in range(max_position_count)
                ]
            )
        return table.get_string()

    # rank 0 且有数据时打印置信度汇总表
    def print_results(self) -> None:
        if dist.get_rank() == 0 and self.rows:
            print("Confidence head reliability metrics:", flush=True)
            print(self.build_table(), flush=True)
