# 训练日志器：模块级单例封装 TensorBoard 与控制台输出。
# 训练循环 init/start_session 后每个 optimizer step 调 on_optimizer_step，
# 指标经 metrics 模块按 logging_steps 窗口累积汇总，主进程写 TB 并打印进度行。
import time  # 会话计时（估算剩余时间）
from typing import Optional  # 可空类型注解

from torch.utils.tensorboard import SummaryWriter  # TensorBoard 标量写入器

from deepspec.utils import ensure_dir, is_global_main_process, print_on_global_main  # 目录创建与全局主进程打印
from deepspec.utils.metrics import add_metric, flush, reset  # 指标记录、汇总与清零


_writer: Optional[SummaryWriter] = None  # TB 写入器单例：仅全局主进程创建
_logging_steps: int = 1  # 每多少个 optimizer step 输出一次日志
_session_start_wall: Optional[float] = None  # 本会话开始的墙钟时间（估剩余时间用）
_session_start_step: int = 0  # 本会话起始全局步数（估剩余时间用）


# 初始化日志器：记录日志间隔；tensorboard_dir 非空且为全局主进程时建目录并开写入器。
def init(*, logging_steps: int, tensorboard_dir: Optional[str] = None) -> None:
    global _writer, _logging_steps  # 修改模块级单例
    _logging_steps = int(logging_steps)  # 日志窗口长度
    if tensorboard_dir is not None and is_global_main_process():  # 只在主进程开 TB（多进程写同一事件文件会损坏）
        ensure_dir(tensorboard_dir)  # 目录不存在则创建
        _writer = SummaryWriter(tensorboard_dir)  # 打开事件文件写入器


# 开启一个训练会话（如断点续训恢复时）：清零指标缓存并记录起始墙钟/步数，供进度估算。
def start_session(*, global_step: int) -> None:  # global_step：会话起始的全局步（恢复点）
    global _session_start_wall, _session_start_step
    reset()  # 丢弃上个会话可能残留的指标
    _session_start_wall = time.time()  # 会话开始时刻
    _session_start_step = int(global_step)  # 会话起始步


# 每个 optimizer step 调用：先记 lr/grad_norm（reduction="last"：窗口内最新值即当前值）；
# 未到日志步只累积不输出；到点 flush 汇总，主进程写 TB 并打印进度行，
# 返回汇总供上层使用（如写入 checkpoint 元数据）。
def on_optimizer_step(
    *,
    global_step: int,  # 当前全局 optimizer step 数
    next_micro_step: int,  # 下一个 micro-batch 序号（推算当前 epoch 用）
    micro_batches_per_epoch: int,  # 每 epoch 的 micro-batch 数
    max_train_steps: int,  # 总训练步数（估剩余时间用）
    learning_rate: float,  # 当前学习率
    grad_norm: float,  # 裁剪前的梯度范数
):
    add_metric("lr", learning_rate, reduction="last", tag="train")  # lr 只关心当前值
    add_metric("grad_norm", grad_norm, reduction="last", tag="train")  # 梯度范数同理

    if global_step % _logging_steps != 0:  # 未到日志窗口末尾
        return None  # 继续累积

    summary = flush()  # 汇总并清空本窗口指标
    if is_global_main_process():  # 只有主进程做输出
        _write_scalars(summary, global_step=global_step)  # 写 TensorBoard
        _print_summary(  # 打印控制台进度行
            summary=summary,
            global_step=global_step,
            next_micro_step=next_micro_step,
            micro_batches_per_epoch=micro_batches_per_epoch,
            max_train_steps=max_train_steps,
        )
    return summary  # 汇总结果交回调用方


# 关闭 TB 写入器并复位单例（训练结束时调用，可安全重复调用）。
def close() -> None:
    global _writer
    if _writer is not None:  # 未初始化则无事可做
        _writer.close()  # 刷出并关闭事件文件
        _writer = None  # 复位，便于再次 init


# 把汇总指标逐条写入 TensorBoard，横轴为全局步；未初始化时静默跳过。
def _write_scalars(summary, *, global_step: int) -> None:  # summary：flush 的汇总结果
    if _writer is None:  # 非主进程或未开 TB
        return
    for key, value in summary.items():  # 逐指标写入
        _writer.add_scalar(key, value, global_step)  # key 形如 "train/loss"


# 打印控制台进度行：epoch/step/loss + 会话已耗时 + 按平均速度估算的剩余时间。
def _print_summary(
    *,
    summary,  # flush 的汇总结果
    global_step: int,  # 当前全局步
    next_micro_step: int,  # 下一个 micro-batch 序号
    micro_batches_per_epoch: int,  # 每 epoch micro-batch 数
    max_train_steps: int,  # 总步数
) -> None:
    session_start_wall = _session_start_wall  # 会话开始时刻
    if session_start_wall is None:  # 未调 start_session 的兜底
        session_start_wall = time.time()  # 以当前时间起算，避免除负
    current_epoch = next_micro_step // micro_batches_per_epoch + 1  # 由 micro 步推算当前 epoch（1 起）
    session_elapsed = time.time() - session_start_wall  # 会话已耗时（秒）
    completed_session_steps = global_step - _session_start_step  # 会话内已完成步数
    remaining_steps = max(max_train_steps - global_step, 0)  # 剩余步数（防越过终点为负）
    remaining_min = (  # 剩余时间 = 已耗时 × 剩余步 / 已完成步（除零用 1 兜底）
        session_elapsed * remaining_steps / max(completed_session_steps, 1)
    ) / 60  # 转分钟
    loss_text = ""  # loss 片段：窗口内无 loss 指标时留空
    if "train/loss" in summary:
        loss_text = f" loss={summary['train/loss']:.4f}"  # 四位小数
    print_on_global_main(  # 主进程打印，自动带时间戳
        f"epoch={current_epoch} "
        f"step={global_step}/{max_train_steps}"
        f"{loss_text} "
        f"| elapsed={session_elapsed / 60:.1f}min"  # 已耗时（分钟，1 位小数）
        f" | remaining={remaining_min:.1f}min"  # 预估剩余时间
    )
