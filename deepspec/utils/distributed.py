# 分布式初始化与协作工具：由环境变量组装全局 rank/world_size 并初始化 NCCL 进程组；
# 另提供主进程判断、带时间戳打印与"主进程先行"屏障上下文。
import contextlib  # contextmanager 装饰器
import os  # 读取 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT 环境变量
import time  # 打印时间戳
from datetime import timedelta  # 进程组超时时长类型

import torch  # GPU 计数、独立随机数生成器
import torch.distributed as dist  # 集合通信（init_process_group/barrier/all_reduce 等）
from torch.utils.data import Sampler  # 采样器基类


# 初始化分布式进程组。假设各节点卡数一致：全局 rank = 节点序号 × 每节点卡数 + 本机卡序号，
# world_size = 节点数 × 每节点卡数；通过 tcp://master_addr:port rendezvous 组网。
def init_dist(local_rank: int, timeout_minutes: int = 60):  # local_rank：本机进程序号；timeout_minutes：集合通信超时（分钟）
    local_world_size = torch.cuda.device_count()  # 本机可见 GPU 数，即每节点进程数
    node_rank = int(os.environ.get("RANK", "0"))  # 当前节点序号（调度器注入；单机默认 0）
    node_world_size = int(os.environ.get("WORLD_SIZE", "1"))  # 总节点数（单机默认 1）
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")  # rendezvous 主节点地址（单机默认本机）
    master_port = os.environ.get("MASTER_PORT", "29500")  # rendezvous 端口
    rank = node_rank * local_world_size + local_rank  # 全局 rank：节点内偏移换算为 0..world_size-1
    world_size = node_world_size * local_world_size  # 全局总进程数
    init_method = f"tcp://{master_addr}:{master_port}"  # TCP rendezvous 地址
    torch.cuda.set_device(local_rank)  # 当前进程绑定对应 GPU（之后所有 CUDA 调用默认用它）
    device = torch.device("cuda", local_rank)  # 本进程的设备对象

    dist.init_process_group(  # 初始化进程组，之后才能做集合通信
        backend="nccl",  # GPU 集合通信后端
        init_method=init_method,  # tcp:// rendezvous
        rank=rank,  # 本进程全局 rank
        world_size=world_size,  # 全局进程数
        timeout=timedelta(minutes=timeout_minutes),  # 超时保护：慢节点不至于无限挂死
        device_id=device,  # 显式指定设备，避免 NCCL 探测与绑定不一致
    )
    return device, rank, world_size  # 供训练器记录拓扑信息


# 是否为全局 rank 0：唯一的"全局主进程"，负责日志、checkpoint 写盘等一次性操作。
def is_global_main_process():
    return dist.get_rank() == 0  # 全局 rank 0


# 是否为本机 0 号卡：节点内主进程，负责本机级别的输出。
def is_local_main_process():
    return torch.cuda.current_device() == 0  # 当前绑定的 GPU 是否 0 号


# 仅全局 rank 0 打印：自动加时间戳并强制 flush，保证多进程日志不重复、实时可见。
def print_on_global_main(*args, **kwargs):  # 透传 print 的位置/关键字参数
    if is_global_main_process():  # 非主进程直接跳过
        kwargs.setdefault("flush", True)  # 立即刷出（挂起/崩溃前最后的日志不能丢）
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, **kwargs)  # 前缀时间戳


# 仅本机 0 号卡打印：节点内每机一条，用于本机相关消息。
def print_on_local_main(*args, **kwargs):
    if is_local_main_process():  # 非 0 号卡跳过
        kwargs.setdefault("flush", True)  # 立即刷出
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, **kwargs)  # 前缀时间戳


# 主进程先行的上下文管理器：rank 0 先进临界区（如下载 tokenizer、建数据缓存），
# 出临界区放屏障；其余 rank 先等屏障再进临界区，从而安全复用 rank 0 的产物。
@contextlib.contextmanager
def main_process_first():
    if dist.get_rank() == 0:  # 主进程：先干活
        yield  # 执行 with 块
        dist.barrier()  # 干完放行其他进程
    else:  # 其余进程：先等
        dist.barrier()  # 阻塞到 rank 0 离开 with 块
        yield  # 主进程产物已就绪，再执行 with 块

# 无状态可续训的确定性分布式采样器：不依赖 dataloader 的 epoch/step 内部状态，
# 只凭 per-rank 已消费偏移即可恢复采样流，跨 epoch 自动换新洗牌继续取数，
# 因此断点续训无需额外保存采样器状态（详细契约见英文 docstring）。
class StatelessResumableDistributedSampler(Sampler):  # 继承 torch 采样器基类
    """Deterministic distributed sampler that streams across epoch boundaries.

    Each epoch uses the first ``total_size`` samples from a deterministic
    shuffle of the dataset.  The sampler can produce a fixed number of
    per-rank samples (``num_samples``) starting from an arbitrary per-rank
    offset, transparently crossing epoch boundaries with fresh shuffles.

    When ``num_samples`` is *None* (default) the sampler yields the remaining
    samples in the current epoch — this preserves backward compatibility with
    code that rebuilds the dataloader at every epoch boundary.
    """

    def __init__(
        self,
        dataset,  # 数据集（只需支持 len()）
        num_replicas: int,  # 并行副本数（通常为全局 world_size）
        rank: int,  # 当前副本的全局 rank
        total_size: int,  # 每 epoch 使用的样本总数（须能被 num_replicas 整除）
        seed: int = 42,  # 洗牌基础种子：第 e 轮种子为 seed + e
        start_global_offset_samples: int = 0,  # 本 rank 已消费的样本数（续训偏移）
        num_samples: int | None = None,  # 本次产出的样本数；None 表示取完当前 epoch 剩余部分
    ):
        assert start_global_offset_samples >= 0, "start_global_offset_samples must be >= 0"  # 偏移非负
        self.dataset = dataset  # 数据集引用
        self.num_replicas = num_replicas  # 副本数
        self.rank = rank  # 本副本 rank
        self.total_size = int(total_size)  # 每 epoch 样本总数
        self.seed = int(seed)  # 洗牌种子
        self.dataset_size = len(self.dataset)  # 数据集总大小（洗牌空间）
        assert self.dataset_size > 0, "dataset must have positive length"  # 空数据集无法采样
        assert self.total_size > 0, "total_size must be > 0"  # 每 epoch 至少 1 条
        assert self.total_size <= self.dataset_size, (  # 每 epoch 截取不能超过数据集本身
            f"total_size ({self.total_size}) cannot exceed dataset size ({self.dataset_size})"
        )
        assert self.total_size % self.num_replicas == 0, (  # 保证各 rank 每 epoch 等长且无 padding
            f"total_size ({self.total_size}) must be divisible by num_replicas ({self.num_replicas})"
        )
        assert num_samples is None or num_samples >= 0, "num_samples must be >= 0"  # 请求步数非负

        self.per_rank_len_per_epoch = self.total_size // self.num_replicas  # 每 rank 每 epoch 的样本数
        self._global_offset = int(start_global_offset_samples)  # per-rank 粒度的全局消费偏移
        self._num_samples = num_samples  # 本次要取的数量（None=取完当前 epoch）

    # 本次迭代将产出的样本数：显式 num_samples 直接用；否则为当前 epoch 剩余量，
    # 偏移恰在 epoch 边界（mod==0）时也返回整 epoch 长度，兼容逐 epoch 重建 dataloader 的用法。
    def __len__(self):
        if self._num_samples is not None:  # 显式指定固定数量
            return self._num_samples
        mod = self._global_offset % self.per_rank_len_per_epoch  # 在当前 epoch 内已消费的位置
        return self.per_rank_len_per_epoch - mod if mod != 0 else self.per_rank_len_per_epoch  # epoch 内剩余量（边界时给整轮）

    # 生成第 epoch_idx 轮的确定性洗牌序列：独立 CPU 生成器 + seed+epoch 种子，
    # 全量洗牌后截前 total_size 个；与 dataloader 状态无关，任意时刻可精确重算。
    def _epoch_perm(self, epoch_idx: int):  # epoch_idx：轮次序号
        g = torch.Generator()  # 独立生成器，不污染全局 RNG
        g.manual_seed(self.seed + epoch_idx)  # 每轮换种子，轮与轮顺序不同
        return torch.randperm(self.dataset_size, generator=g).tolist()[: self.total_size]  # 洗牌并截取

    # 本 rank 在该轮的样本切片：以 rank 为起点、num_replicas 为步长跨步取索引，
    # 各 rank 得到互不重叠且等长的样本。
    def _epoch_slice_for_rank(self, perm):  # perm：该轮完整洗牌序列
        return perm[self.rank : self.total_size : self.num_replicas]  # 跨步切片实现无重叠等分

    # 无限采样流生成器：从当前偏移的恢复点继续产出索引；epoch 内取完自动进入
    # 下一轮（新洗牌）无限继续，实际产出量由 __iter__ 按 __len__ 截断。
    def _iter_stream(self):
        epoch_idx = self._global_offset // self.per_rank_len_per_epoch  # 偏移折算出当前轮次
        offset_in_epoch = self._global_offset % self.per_rank_len_per_epoch  # 本轮内已消费数

        perm = self._epoch_perm(epoch_idx)  # 当前轮洗牌序列
        my_seq = self._epoch_slice_for_rank(perm)  # 本 rank 切片
        for i in range(offset_in_epoch, len(my_seq)):  # 从恢复点续取完本轮
            yield my_seq[i]

        epoch_idx += 1  # 进入下一轮
        while True:  # 无限流：后续轮全量输出
            perm = self._epoch_perm(epoch_idx)  # 新轮新洗牌
            my_seq = self._epoch_slice_for_rank(perm)  # 本 rank 切片
            for idx in my_seq:  # 整轮产出
                yield idx
            epoch_idx += 1  # 再下一轮

    # 对外迭代入口：从无限流中取 __len__ 个索引后停止，
    # 因此续训只需把 start_global_offset_samples 设为已消费数即可无缝接上。
    def __iter__(self):
        it = self._iter_stream()  # 创建无限流
        for _ in range(len(self)):  # 只产出本次需要的数量
            yield next(it)  # 逐个取流中下一个索引
