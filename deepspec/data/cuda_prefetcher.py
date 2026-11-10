from threading import Thread  # 后台线程：让"取下一批 + H2D 拷贝"与当前批的前向/反向计算重叠

import torch  # PyTorch：CUDA 流管理、张量设备搬运与 dtype 转换


def move_batch_to_device(batch, device):
    # 把 CPU 批次逐键搬到 device：non_blocking=True 依赖 pin_memory 锁页内存，
    # 使 H2D 拷贝真正异步，从而与侧流/计算流重叠而不是阻塞主线程。
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    # Embedding lookup requires int64; cast on GPU to avoid bloating CPU-to-GPU transfer.
    # 在 GPU 上补一次 long 转换：若 DataLoader 侧给出 int32 等窄 dtype，先窄带宽传上来再就地转 int64，
    # 比在 CPU 转好再传（8 字节/元素）节省一半以上 PCIe 带宽；embedding 查表要求 int64 索引。
    if moved["input_ids"].dtype != torch.long:
        moved["input_ids"] = moved["input_ids"].to(torch.long)
    return moved


class CUDAPrefetcher:
    """Overlaps DataLoader iteration and H2D transfer with compute.

    Uses a background thread and a dedicated CUDA stream so that both
    next(dataloader) and the H2D copy for the next batch run concurrently
    with the forward/backward of the current batch.
    """

    # 预取器原理：数据加载与 H2D 拷贝是 CPU/PCIe 瓶颈，若串行执行 GPU 会空等。
    # 这里用"专用 CUDA 流 + 后台线程"把下一批的获取与上传藏进当前批的计算窗口：
    # CPU 侧 next(dataloader) 在后台线程跑（GIL 在 IO/解压时释放），GPU 侧 H2D 走 side stream，
    # 两条流水线与 forward/backward 并行，消除逐批的传输气泡。

    def __init__(self, dataloader, device):
        self.dataloader = dataloader  # 被包装的 DataLoader（自身需 pin_memory=True 才能异步 H2D）
        self.device = device  # 目标 CUDA 设备（如 torch.device("cuda", local_rank)）
        self.stream = torch.cuda.Stream(device=device)  # 专属侧流：H2D 拷贝在其上异步排队执行

    def __iter__(self):
        # 每个 epoch 重建底层迭代器并清空预取状态；for 循环每次进入都会调用 __iter__。
        self._iter = iter(self.dataloader)  # 底层 DataLoader 迭代器（worker 进程从此开始拉数据）
        self._done = False  # 底层迭代是否已耗尽（StopIteration 标记）
        self._gpu_batch = None  # 已完成 H2D 排队（未必完成拷贝）的下一批，缓存到 GPU
        self._thread = None  # 正在预取的后台线程句柄；None 表示当前无在途预取
        # First batch: fetch synchronously so __next__ has something to return.
        # 首批同步取：迭代开始时没有可 join 的线程，必须先产出一份 GPU 批供 __next__ 返回。
        self._fetch_and_transfer()
        return self

    def _fetch_and_transfer(self):
        """Pop the next CPU batch from the DataLoader and queue H2D on the side stream."""
        # 从 DataLoader 取下一个 CPU 批，并在侧流上排队 H2D 拷贝；IO 等待与拷贝
        # 都发生在后台线程/侧流，不阻塞当前批的前向/反向。
        try:
            cpu_batch = next(self._iter)  # 阻塞拉取下一 CPU 批（worker 预取不足时在此等待）
        except StopIteration:
            self._done = True  # 数据耗尽：置标记，__next__ 下次到货后抛 StopIteration 结束循环
            return
        with torch.cuda.stream(self.stream):  # 进入侧流上下文：下面的 .to(device) 排队到该流
            self._gpu_batch = move_batch_to_device(cpu_batch, self.device)

    def __next__(self):
        # 迭代协议：join 上一轮在途预取 → 等侧流拷贝完成 → 记录流依赖 → 再启动下一轮预取。
        # Join the background thread kicked off in the previous __next__.
        if self._thread is not None:
            self._thread.join()  # 等后台线程结束：保证 _gpu_batch 的 H2D 已全部排队、无并发写
            self._thread = None

        if self._done:
            raise StopIteration  # 正常结束 for 循环

        # Make the compute stream wait until the side-stream H2D is complete.
        current = torch.cuda.current_stream(self.device)  # 当前计算流（forward/backward 所在流）
        current.wait_stream(self.stream)  # 流间同步：计算流等侧流的 H2D 全部完成（事件等待，非 CPU 阻塞）
        batch = self._gpu_batch

        # Prevent the caching allocator from recycling these tensors before
        # the compute stream is done with them.
        # 跨流张量的生命周期保护：wait_stream 只是流间约束，缓存分配器仍可能把显存
        # 归还给其他流复用；record_stream 让分配器知道这些张量在计算流上还会被使用，
        # 直到该流事件完成后才允许回收，避免读写竞态损坏数据。
        for value in batch.values():
            value.record_stream(current)

        # Kick off the next fetch and H2D in a background thread so it
        # overlaps with compute on the batch we are about to return.
        # 立即异步启动下一批预取：DataLoader 拉取 + 侧流 H2D 与本批的前向/反向并行，
        # 返回 batch 后调用方即可开算，预取在幕后继续。
        self._thread = Thread(target=self._fetch_and_transfer, daemon=True)
        self._thread.start()

        return batch

    def __len__(self):
        # 透传底层 epoch 的批数，便于外部计算进度/日志；与预取机制无关。
        return len(self.dataloader)
