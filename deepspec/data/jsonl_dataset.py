import hashlib  # blake2b：对"路径+mtime"指纹哈希，生成行索引缓存文件名
import json  # 解析 JSONL 每行为 Python 对象（对话样本）
import mmap  # 内存映射文件：随机读 JSONL 行时避免 read 系统调用与用户态缓冲拷贝
import os  # 路径处理、目录创建、fsync
import pickle  # 序列化行偏移索引到磁盘缓存（多进程 DataLoader 共享只读）
from bisect import bisect_right  # 二分查找：全局样本号 → (文件号, 文件内行号) 映射
import torch  # 仅用作 Dataset 基类载体
from tqdm import tqdm  # 构建行索引时的进度条
from deepspec.utils.constant import CACHE_DIR  # 行索引缓存根目录（约定于全局配置）

class JsonLineDataset(torch.utils.data.Dataset):
    # 基于 mmap 的 JSONL 数据集：把每个文件的"每行起始偏移"预扫描成索引并落盘缓存。
    # __getitem__ 只需 seek 到偏移再读一行 JSON，O(1) 随机访问且不加载整个文件；
    # 多个文件拼接为一个逻辑数据集，用前缀和 + 二分定位全局 idx 属于哪个文件。
    # 典型用途：训练前按路径列出若干 JSONL（每行一个 conversations 样本）。
    def __init__(self, data_paths):
        super().__init__()
        self.data_paths = sorted(data_paths)  # 排序保证多进程/多节点构建出的缓存键与顺序一致
        self.cache_dir = os.path.abspath(CACHE_DIR)  # 索引缓存目录（绝对路径，避免受 cwd 影响）
        os.makedirs(self.cache_dir, exist_ok=True)  # 缓存目录不存在则创建（幂等）
        self.num_data_per_file = []  # 每个文件的行数（样本数），构建索引时逐文件追加
        self.cum_counts = [0]  # 文件行数前缀和：cum_counts[i] = 前 i 个文件的累计样本数
        self.files = [None] * len(self.data_paths)  # 各文件的二进制句柄，惰性打开（None=未打开）
        self.mmaps = [None] * len(self.data_paths)  # 各文件的 mmap 对象，惰性建立
        self.line_starts_per_file = [None] * len(self.data_paths)  # 各文件每行起始字节偏移列表
        self._build_all_line_starts()  # 扫描全部文件建立行偏移索引（命中缓存则直接读 pickle）
        for count in self.num_data_per_file:  # 累计各文件样本数 → 全局前缀和表
            self.cum_counts.append(self.cum_counts[-1] + count)
        self.num_data = self.cum_counts[-1]  # 数据集总样本数（所有文件行数之和）
        self.close()  # 构建期临时打开的句柄/mmap 全部关闭；后续按需惰性重开，省 fd

    def __len__(self):
        # Dataset 协议：样本总数，供 sampler/dataloader 计算轮次与进度。
        return self.num_data

    def __getitem__(self, idx):
        # 随机访问第 idx 个样本：定位文件 → 惰性 mmap → seek 行偏移 → 读一行并反序列化。
        if not (0 <= idx < self.num_data):
            raise IndexError(idx)  # 越界显式报错，避免静默映射到错误样本
        file_idx, local_idx = self._map_global_to_local(idx)  # 全局下标 → (文件号, 行号)
        if self.files[file_idx] is None or self.mmaps[file_idx] is None:
            self._open_file_mmap(file_idx)  # 该文件尚未 mmap（如上一轮被 close）则先打开
        mm = self.mmaps[file_idx]  # 当前文件的内存映射视图
        starts = self.line_starts_per_file[file_idx]  # 该文件的行起始偏移索引
        mm.seek(starts[local_idx])  # 直接跳到目标行：mmap 下 seek+readline 只触发所需页的缺页
        line = mm.readline().decode("utf-8")  # 读到行尾（含换行符），解码为 str
        return json.loads(line)  # 解析为 dict（约定含 "conversations" 字段的对话样本）

    def close(self):
        # 释放全部 mmap 与文件句柄（幂等）：构建结束、数据集淘汰、进程退出时调用。
        # mmap 句柄持有文件锁页映射，长期持有会占虚拟地址空间并阻止文件被替换。
        for idx, mm in enumerate(self.mmaps):
            if mm is not None:
                mm.close()  # 解除内存映射
                self.mmaps[idx] = None  # 置空以便惰性重开（__getitem__ 会检测 None）
        for idx, handle in enumerate(self.files):
            if handle is not None:
                handle.close()  # 关闭底层文件描述符
                self.files[idx] = None

    def __del__(self):  # pragma: no cover
        # 析构兜底：对象被回收时确保句柄关闭，防止 fd 泄漏（多 worker 反复建数据集时尤为重要）。
        self.close()

    def _map_global_to_local(self, global_idx):
        # 全局样本号 → (文件号, 文件内行号)：在前缀和数组里二分。
        # bisect_right 找到第一个 cum_counts > global_idx 的位置再减一，
        # 即 global_idx 落入的文件（前闭后开区间 [cum[i], cum[i+1]) 语义正确）。
        file_idx = bisect_right(self.cum_counts, global_idx) - 1
        local_idx = global_idx - self.cum_counts[file_idx]  # 减去该文件起始计数得局部行号
        return file_idx, local_idx

    def _open_file_mmap(self, index):
        # 惰性打开第 index 个文件并建立只读 mmap（close 后再次访问时走此路径）。
        if self.files[index] is None:
            self.files[index] = open(self.data_paths[index], "rb")  # 二进制读，mmap 需要 fd
        if self.mmaps[index] is None:
            self.mmaps[index] = mmap.mmap(  # 只读映射：数据集内容被视为不可变
                self.files[index].fileno(), 0, access=mmap.ACCESS_READ  # fd；length=0 映射整个文件
            )

    def _file_key(self, path):
        # 生成缓存失效指纹："归一化绝对路径 | 纳秒级 mtime"。
        # 文件内容一旦被重写（mtime 变化）即换新缓存键，避免读到过期的行偏移索引。
        abspath = os.path.abspath(path)
        st = os.stat(abspath)
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))  # 优先纳秒精度，旧平台回退浮点秒
        return f"{os.path.normcase(abspath)}|{mtime_ns}"  # normcase 统一大小写/分隔符（Windows 兼容）

    def _cache_path_from_key(self, file_key):
        # 由指纹派生缓存文件路径：blake2b 128bit 摘要，避免路径里非法字符与超长文件名。
        key_hash = hashlib.blake2b(file_key.encode("utf-8"), digest_size=16).hexdigest()
        return os.path.join(self.cache_dir, f"jsonlindex-{key_hash}.pkl")

    def _atomic_pickle_dump(self, obj, dst_path):
        # 原子写 pickle：先写临时文件（带 pid 防并发冲突）并 fsync 落盘，
        # 再 os.replace 原子改名。多进程同时构建同一索引时，读者要么看到完整旧缓存、
        # 要么看到完整新缓存，绝不会读到写了一半的损坏 pickle。
        tmp_path = f"{dst_path}.tmp-{os.getpid()}"
        with open(tmp_path, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)  # 最高协议：最快且支持大对象
            handle.flush()  # 刷用户态缓冲
            os.fsync(handle.fileno())  # 强制内核写回磁盘：即使随后断电，rename 后的文件也完整
        os.replace(tmp_path, dst_path)  # 同目录原子改名（POSIX 保证原子性与可见性）

    def _build_all_line_starts(self):
        # 核心索引构建：逐文件确定每行起始偏移；优先读磁盘缓存，未命中则实际扫描文件并写回缓存。
        # 行偏移只依赖文件字节布局，与内容语义无关，因此可以安全持久化复用。
        for idx, path in tqdm(enumerate(self.data_paths), total=len(self.data_paths)):
            file_key = self._file_key(path)  # 当前文件的失效指纹
            cache_path = self._cache_path_from_key(file_key)  # 对应的索引缓存路径
            if cache_path is not None and os.path.exists(cache_path):
                with open(cache_path, "rb") as handle:
                    cached = pickle.load(handle)  # 反序列化 {file_key, file_path, line_starts}
                if (
                    isinstance(cached, dict)
                    and cached.get("file_key") == file_key  # 双重校验指纹，防止哈希碰撞/文件被改动
                    and isinstance(cached.get("line_starts"), list)
                ):
                    starts = cached["line_starts"]  # 直接复用缓存的行偏移，免去全文件扫描
                    self.line_starts_per_file[idx] = starts
                    self.num_data_per_file.append(len(starts))  # 行数即样本数
                    continue

            handle = open(path, "rb")  # 缓存未命中：实际扫描文件
            mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self.files[idx] = handle  # 扫描期间暂存句柄，供 readline 使用
            self.mmaps[idx] = mm
            starts = []  # 行起始偏移数组（第一个元素恒为 0）
            mm.seek(0)
            pos = 0
            while True:
                starts.append(pos)  # 记录当前行起始位置
                line = mm.readline()  # 逐行推进（mmap readline 无额外拷贝）
                if not line:
                    break  # readline 返回 b"" 表示到达文件尾
                pos = mm.tell()  # tell 给出映射内当前字节偏移，即下一行起点
            if starts and mm.size() == pos:
                starts.pop()  # 文件以换行结尾时最后一次 readline 指向文件尾，不是真行，剔除
            self.line_starts_per_file[idx] = starts  # 保存该文件索引
            self.num_data_per_file.append(len(starts))
            if cache_path is not None:
                self._atomic_pickle_dump(  # 写回缓存供下次启动与多进程复用
                    {
                        "file_key": file_key,  # 记录指纹供加载时校验
                        "file_path": os.path.abspath(path),  # 仅供调试溯源
                        "line_starts": starts,  # 行偏移列表
                    },
                    cache_path,
                )
