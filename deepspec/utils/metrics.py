# 训练指标统计：训练循环用 add_metric 累积标量/比率指标，flush 在日志窗口末尾做
# 跨 DP 卡归约并输出汇总，reset 清空缓存。比率指标（如接受率）必须以分子/分母形式
# 上报，全局比率 = sum(分子)/sum(分母)，比"先局部平均再平均"更精确。
import re  # 校验 reduction 名的合法性

import torch  # 张量堆叠/求和/集合通信
import torch.distributed as dist  # DP 卡间 all_reduce/all_gather


_REDUCTION_PATTERN = re.compile(r"^(dp_)?(mean|sum|max|min|last)$")  # 合法归约名：可带 dp_ 前缀表示需跨卡归约
_DEFAULT_RATIO_REDUCTION = "dp_sum"  # 比率指标默认归约：全局 sum/sum
_metrics = {}  # 指标缓存："{tag}/{name}" -> {"kind","reduction", 累积列表}


# 输入标准化为 0 维标量张量：张量先 detach（阻断 autograd 引用）并断言单元素；
# Python 数值转 fp32 张量，统一后续堆叠/归约处理。
def _detach_scalar(value):  # value：标量张量或数值
    if torch.is_tensor(value):  # 张量输入：脱离计算图
        value = value.detach()  # 脱钩：防止指标缓存持有反向图导致显存泄漏
        assert value.numel() == 1, "metrics only support scalar values"  # 仅支持标量
        return value.reshape(())  # 摊平成 0 维
    return torch.tensor(float(value), dtype=torch.float32)  # 数值统一转 fp32 张量


# 复制一份用于集合通信：detach+clone 与原张量生命周期解耦并统一 fp32；
# NCCL 后端要求张量在 GPU 上，CPU 张量先搬到当前卡。
def _clone_to_reduce_device(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().clone().to(torch.float32)  # 独立副本：通信会就地修改，不能动原数据
    if dist.get_backend() == "nccl" and not tensor.is_cuda:  # NCCL 只支持 CUDA 张量
        tensor = tensor.to(torch.device("cuda", torch.cuda.current_device()))  # 搬到当前进程 GPU
    return tensor  # 通信就绪的副本


# 跨全部 DP 卡归约一个 0 维张量：sum/mean/max/min 直接 all_reduce（就地写回）；
# "last" 无合并语义，只能 all_gather 后取一个代表（最高 rank 的值）。
def _reduce_dp_value(value: torch.Tensor, op_name: str) -> torch.Tensor:  # op_name：sum/mean/max/min/last
    if op_name == "sum" or op_name == "mean":  # 求和/求均值类
        dist.all_reduce(value, op=dist.ReduceOp.SUM)  # 各卡求和，结果写回 value
        if op_name == "mean":  # 均值 = 总和 / 卡数
            value = value / dist.get_world_size()  # 除以全局卡数
        return value
    if op_name == "max":  # 取最大
        dist.all_reduce(value, op=dist.ReduceOp.MAX)  # 全局最大
        return value
    if op_name == "min":  # 取最小
        dist.all_reduce(value, op=dist.ReduceOp.MIN)  # 全局最小
        return value
    if op_name == "last":  # "取一个代表值"场景
        gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]  # 每卡一个接收缓冲
        dist.all_gather(gathered, value)  # 收集所有卡的值
        return gathered[-1]  # 取最高 rank 的值
    raise AssertionError(f"unsupported reduction: {op_name}")  # 理论不可达（入口已用正则校验）


# 窗口内本卡累积值的局部归约（尚未跨卡）：last 取最近一条；max/min 取极值；
# mean/sum 都实现为窗口内平均——sum 语义指标（如每步 token 数）在窗口内
# 逐条记录总量，平均即"每条记录的平均量"（见 flush 中 dp_ 前缀的跨卡求和配合）。
def _local_reduce(values, reduction: str) -> torch.Tensor:  # values：窗口内累积的张量列表
    assert values, "cannot reduce an empty metric buffer"  # 空缓存说明窗口内没记录过
    if reduction == "last":  # 最近一次记录，无需堆叠
        return values[-1]
    stacked = torch.stack([_clone_to_reduce_device(value) for value in values])  # 堆成 [记录数] 的 fp32 张量
    if reduction == "max":  # 窗口内最大
        return stacked.max()
    if reduction == "min":  # 窗口内最小
        return stacked.min()
    if reduction in ("mean", "sum"):  # 平均类（sum 也按窗口平均上报）
        # Sum-style metrics report the average per emit in the logging window.
        return stacked.mean()  # 窗口内逐条平均
    raise AssertionError(f"unsupported reduction: {reduction}")  # 理论不可达


# 汇总当前指标缓存的"模式"：(名字, 类型, 归约, 条数) 列表，用于跨 rank 一致性校验。
def _schema():
    items = []  # 模式描述列表
    for name, entry in sorted(_metrics.items()):  # 按名字排序，保证跨 rank 顺序可比
        if entry["kind"] == "ratio":  # 比率指标以分子条数为准
            count = len(entry["num"])
        else:  # 标量指标看值条数
            count = len(entry["values"])
        items.append((name, entry["kind"], entry["reduction"], count))  # 一条模式记录
    return items


# 校验所有 rank 的指标模式完全一致：add_metric 是各 rank 独立调用的，若某 rank
# 少记/多记/改名，flush 的集合通信会错位甚至死锁，这里尽早断言暴露问题。
def _assert_schema_consistent():
    local_schema = _schema()  # 本卡模式
    gathered = [None for _ in range(dist.get_world_size())]  # 接收各卡模式
    dist.all_gather_object(gathered, local_schema)  # gather 任意 Python 对象（模式很小，开销可忽略）
    reference = gathered[0]  # 以 rank 0 为基准
    for rank, schema in enumerate(gathered[1:], start=1):  # 逐一比对
        assert schema == reference, (  # 不一致立即报错并打印双方模式
            "metric schema mismatch across ranks: "
            f"rank0={reference}, rank{rank}={schema}"
        )


# 安全除法：分母为 0（窗口内无有效样本）时返回 0.0，避免 NaN 写进日志。
def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if denominator.item() == 0:  # 分母为 0
        return 0.0
    return (numerator / denominator).item()  # 全局比率转 Python float


# 记录一条指标（禁用 torch.compile：指标记录是纯簿记，编译反而引入图断裂与开销；
# recursive=False 只禁用本函数，不影响内部调用的其他函数）。
# 契约详见英文 docstring：den= 传分母即为"比率指标"，勿在本地预先相除，
# flush 以全局 sum(num)/sum(den) 计算精确比率；同名指标的 kind/reduction 不可中途改变。
@torch.compiler.disable(recursive=False)
def add_metric(
    name,  # 指标名（如 "loss"），实际缓存键为 "{tag}/{name}"
    value,  # 标量值：张量或 Python 数值
    *,  # 以下参数仅限关键字
    den=None,  # 比率指标的分母；None 表示普通标量指标
    reduction: str = _DEFAULT_RATIO_REDUCTION,  # 归约方式；比率指标只允许默认 dp_sum
    tag: str = "train",  # 命名空间前缀（train/eval 等）
):
    """Record one scalar metric for the next logging-window flush.

    Tensor inputs are detached at the API boundary to prevent metric logging
    from retaining autograd graphs. Ratios must pass their denominator through
    ``den=``; callers should not pre-divide locally because ``flush`` computes
    the global ratio as ``sum(num) / sum(den)``.
    """

    assert _REDUCTION_PATTERN.match(reduction), f"unsupported reduction: {reduction}"  # 归约名合法性
    metric_name = f"{tag}/{name}"  # 全名：如 train/loss
    if den is not None:  # 带分母 → 比率指标
        assert reduction == _DEFAULT_RATIO_REDUCTION, (  # 比率只支持 dp_sum，保证全局 sum/sum 语义
            "ratio metrics must use the default dp_sum reduction"
        )
        value = _detach_scalar(value)  # 分子标准化
        den = _detach_scalar(den)  # 分母标准化
        entry = _metrics.setdefault(  # 首次记录时创建缓存条目
            metric_name,
            {"kind": "ratio", "reduction": reduction, "num": [], "den": []},  # 比率条目：分子/分母各自累积
        )
        assert entry["kind"] == "ratio", f"metric kind changed for {metric_name}"  # 同名指标类型不可中途改变
        assert entry["reduction"] == reduction, (  # 同名指标归约方式不可中途改变
            f"metric reduction changed for {metric_name}: "
            f"{entry['reduction']} != {reduction}"
        )
        entry["num"].append(value)  # 累积分子
        entry["den"].append(den)  # 累积分母
        return  # 比率在 flush 时统一计算，无需更多处理

    value = _detach_scalar(value)  # 标量指标同样先标准化
    entry = _metrics.setdefault(  # 首次记录时创建缓存条目
        metric_name,
        {"kind": "scalar", "reduction": reduction, "values": []},  # 标量条目：只累积值列表
    )
    assert entry["kind"] == "scalar", f"metric kind changed for {metric_name}"  # 类型一致性
    assert entry["reduction"] == reduction, (  # 归约方式一致性
        f"metric reduction changed for {metric_name}: "
        f"{entry['reduction']} != {reduction}"
    )
    entry["values"].append(value)  # 累积本窗口的第 N 条记录


# 窗口末尾汇总：先校验跨 rank 模式一致，再逐指标归约。
# 比率指标：分子、分母各自跨卡求和后相除，得到全局精确比率（如全局接受率）；
# 标量指标：先窗口内局部归约，dp_ 前缀的再用同名算子跨卡归约。
# finally 中 reset：即使归约中途异常也清空缓存，避免下个窗口残留旧数据。
def flush() -> dict[str, float]:  # 返回 {指标名: float}
    _assert_schema_consistent()  # 集合通信的前置条件检查
    try:  # finally 保证 reset 一定执行
        summary = {}  # 汇总结果
        for name, entry in sorted(_metrics.items()):  # 按名字序处理，日志列序稳定
            reduction = entry["reduction"]  # 该指标声明的归约方式
            if entry["kind"] == "ratio":  # 比率指标：全局 sum/sum
                num = torch.stack(  # 分子堆叠
                    [_clone_to_reduce_device(value) for value in entry["num"]]  # 窗口内全部记录（已就通信设备）
                ).sum()  # 本卡分子总和
                den = torch.stack(  # 分母同理
                    [_clone_to_reduce_device(value) for value in entry["den"]]  # 分母记录
                ).sum()  # 本卡分母总和
                num = _reduce_dp_value(num, "sum")  # 分子跨卡求和
                den = _reduce_dp_value(den, "sum")  # 分母跨卡求和
                summary[name] = _safe_div(num, den)  # 全局比率（分母 0 保护）
                continue  # 比率处理完毕

            local_reduction = reduction[3:] if reduction.startswith("dp_") else reduction  # 剥掉 dp_ 前缀得局部归约名
            value = _local_reduce(entry["values"], local_reduction)  # 窗口内本卡归约
            if reduction.startswith("dp_"):  # 需要跨卡时再全局归约
                value = _reduce_dp_value(value, local_reduction)  # 用同名算子跨卡归约
            summary[name] = value.item()  # 转 Python float
        return summary  # 本窗口汇总
    finally:
        reset()  # 无论成败都清空缓存


# 清空指标缓存（flush 后自动调用；训练会话切换时可手动调用）。
def reset() -> None:
    _metrics.clear()  # 直接清字典，条目对象随之释放


__all__ = ["add_metric", "flush", "reset"]  # 对外只暴露记录/汇总/清零三个入口
