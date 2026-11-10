"""Target-cache storage protocol, writers, dataset, collator, and validation."""

# 目标缓存（target cache）体系：为 draft model（EAGLE3/DSpark）训练预物化目标模型的隐状态。
# 背景：投机解码的 draft 训练需要目标模型中间层与末层 hidden states 作为监督信号；
# 若训练时在线跑目标模型前向，显存与算力开销巨大。本模块把这些信号离线导出为
# 二进制分片（shard-*.bin），配固定 56 字节定长索引（samples.idx）与 manifest.json 元信息。
# 数十 TB 级存储正是由此而来：单样本 hidden states ≈ seq_len × num_target_layers × hidden
# × 2 字节（bfloat16），百万级长样本即达 38TB 量级，必须分片存储 + mmap 随机读取。

import json  # manifest.json 与 summary.json 的读写
import mmap  # 索引文件与 shard 二进制的只读内存映射，避免随机读的 read 系统调用开销
import os  # 路径拼接、文件存在性检查、fsync、原子改名
import queue  # 异步 writer 的有界队列（生产者-消费者）
import shutil  # 清理 _tmp 临时目录
import struct  # 定长二进制索引记录的打包/解包（协议格式固定，跨进程稳定）
import threading  # 异步 writer 的后台落盘线程
from collections import OrderedDict  # shard 句柄的 LRU 淘汰（记录访问顺序）
from dataclasses import dataclass  # 声明式数据类
from typing import Dict, List  # 类型注解

import numpy as np  # frombuffer：零拷贝视图读二进制，再 copy 出独立数组
import torch  # 张量构造与 dtype 转换

from deepspec.data.parser import preprocess_record  # 原始对话样本的 tokenize + loss_mask 构造


TARGET_CACHE_VERSION = 2  # 存储协议版本号：格式变更时递增，读取端据此拒绝旧格式缓存
# 索引记录布局（小端）：sample_id(Q8) shard_id(I4) seq_len(I4) + 五个段内偏移(Q8×5) = 56 字节。
# 定长记录支持 O(1) 随机寻址：第 i 个样本的记录在 samples.idx 的 i*56 字节处。
INDEX_RECORD_STRUCT = struct.Struct("<QIIQQQQQ")
INDEX_RECORD_SIZE = INDEX_RECORD_STRUCT.size  # 56：写读两端与校验逻辑共用的记录字节数

TARGET_CACHE_HIDDEN_DTYPE = "bfloat16"  # 隐状态存储 dtype：bf16 省一半带宽/容量，精度足够做监督
TARGET_CACHE_TOKEN_DTYPE  = "int32"    # input_ids 存 int32（词表 id 远小于 2^31，省一半 vs int64）
TARGET_CACHE_MASK_DTYPE   = "uint8"    # mask 类张量存 uint8（0/1 值，最小存储）

# 各 dtype 的元素字节数体现在 expected_*_nbytes 的乘数里：int32=4、uint8=1、bf16=2。


def atomic_json_dump(payload, path: str):
    # 原子写 JSON：先写 .tmp 并 fsync，再 os.replace 原子改名。
    # 保证崩溃/并发下读者看到的 manifest 永远是完整一致的版本。
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)  # 排序键：输出 diff 友好、字节级确定
        handle.flush()
        os.fsync(handle.fileno())  # 强制落盘：rename 后即使断电文件也完整
    os.replace(tmp_path, path)  # POSIX 原子改名：要么旧文件要么新文件，无中间态


def build_target_cache_shard_path(cache_dir: str, file_name: str) -> str:
    # 由缓存目录与 shard 文件名拼出 shard 路径；路径规则集中于这一处，避免各处手拼不一致。
    return os.path.join(cache_dir, file_name)


def expected_target_cache_tensor_numel(
    *,
    seq_len: int,  # 样本 token 数
    hidden_size: int,  # 目标模型隐状态维度
    num_target_layers: int,  # 监督用的目标模型层数（EAGLE3 取若干中间层）
):
    # 给定样本形状，计算五个张量的元素总数；是磁盘布局的"逻辑视图"。
    return {
        "input_ids": int(seq_len),
        "attention_mask": int(seq_len),
        "loss_mask": int(seq_len),
        "target_hidden_states": int(seq_len) * int(num_target_layers) * int(hidden_size),  # 各中间层拼接展平
        "target_last_hidden_states": int(seq_len) * int(hidden_size),  # 末层隐状态
    }


def expected_target_cache_tensor_nbytes(
    *,
    seq_len: int,
    hidden_size: int,
    num_target_layers: int,
):
    # 元素数 × 元素字节数 = 各段字节长度：int32×4、uint8×1、bf16×2。
    # 读取端据此做越界校验，写入端据此计算 shard 容量。
    numel = expected_target_cache_tensor_numel(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
    )
    return {
        "input_ids": numel["input_ids"] * 4,  # int32
        "attention_mask": numel["attention_mask"],  # uint8
        "loss_mask": numel["loss_mask"],  # uint8
        "target_hidden_states": numel["target_hidden_states"] * 2,  # bfloat16
        "target_last_hidden_states": numel["target_last_hidden_states"] * 2,  # bfloat16
    }


def pack_index_record(
    *,
    sample_id: int,  # 全局样本号（0 起，稠密递增）
    shard_id: int,  # 样本所在 shard 的全局编号
    seq_len: int,  # 该样本 token 数
    input_ids_offset: int,  # input_ids 段在 shard 文件内的字节偏移
    attention_mask_offset: int,
    loss_mask_offset: int,
    target_hidden_states_offset: int,
    target_last_hidden_states_offset: int,
):
    # 把一条索引记录按协议打包成 56 字节：offset 用 Q（u64）以支持超大单文件（>4GB）分片。
    return INDEX_RECORD_STRUCT.pack(
        int(sample_id),
        int(shard_id),
        int(seq_len),
        int(input_ids_offset),
        int(attention_mask_offset),
        int(loss_mask_offset),
        int(target_hidden_states_offset),
        int(target_last_hidden_states_offset),
    )


def unpack_index_record(buffer, offset: int = 0):
    # 从缓冲（通常为 samples.idx 的 mmap）按偏移解出一条索引记录，返回语义化 dict。
    (
        sample_id,
        shard_id,
        seq_len,
        input_ids_offset,
        attention_mask_offset,
        loss_mask_offset,
        target_hidden_states_offset,
        target_last_hidden_states_offset,
    ) = INDEX_RECORD_STRUCT.unpack_from(buffer, offset)
    return {
        "sample_id": sample_id,
        "shard_id": shard_id,
        "seq_len": seq_len,
        "input_ids_offset": input_ids_offset,
        "attention_mask_offset": attention_mask_offset,
        "loss_mask_offset": loss_mask_offset,
        "target_hidden_states_offset": target_hidden_states_offset,
        "target_last_hidden_states_offset": target_last_hidden_states_offset,
    }


def load_target_cache_manifest(cache_dir: str):
    # 读取并完整校验 manifest.json：格式不符立即断言失败，防止静默读坏缓存。
    manifest_path = os.path.join(cache_dir, "manifest.json")
    assert os.path.exists(manifest_path), f"Missing target cache manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_target_cache_manifest(cache_dir=cache_dir, manifest=manifest)
    return manifest


def validate_target_cache_manifest(*, cache_dir: str, manifest):
    # manifest 全面体检：字段齐全、版本/dtype/记录大小匹配协议、shard 连续且文件存在、
    # samples.idx 大小与 num_samples 严格一致。训练启动时执行一次，把错误拦在最前面。
    required_fields = {
        "version",
        "num_samples",
        "num_shards",
        "target_layer_ids",
        "hidden_dtype",
        "token_dtype",
        "mask_dtype",
        "index_record_size",
        "hidden_size",
        "shards",
    }
    missing = sorted(required_fields - set(manifest))  # 缺失字段列表（排序保证报错信息稳定）
    assert not missing, (
        f"Target cache manifest is missing required fields {missing}: "
        f"{os.path.join(cache_dir, 'manifest.json')}"
    )
    assert int(manifest["version"]) == TARGET_CACHE_VERSION, (  # 版本不匹配即拒绝（协议不兼容）
        "Unsupported target cache manifest version: "
        f"{manifest['version']} != {TARGET_CACHE_VERSION}"
    )
    assert manifest["hidden_dtype"] == TARGET_CACHE_HIDDEN_DTYPE, (  # 隐状态 dtype 必须是 bf16
        "Unsupported hidden_dtype in target cache manifest: "
        f"{manifest['hidden_dtype']}"
    )
    assert manifest["token_dtype"] == TARGET_CACHE_TOKEN_DTYPE, (  # token dtype 必须是 int32
        "Unsupported token_dtype in target cache manifest: "
        f"{manifest['token_dtype']}"
    )
    assert manifest["mask_dtype"] == TARGET_CACHE_MASK_DTYPE, (  # mask dtype 必须是 uint8
        "Unsupported mask_dtype in target cache manifest: "
        f"{manifest['mask_dtype']}"
    )
    assert int(manifest["index_record_size"]) == INDEX_RECORD_SIZE, (  # 记录大小与代码内协议一致
        "index_record_size does not match canonical target cache protocol: "
        f"{manifest['index_record_size']} != {INDEX_RECORD_SIZE}"
    )
    hidden_size = int(manifest["hidden_size"])
    assert hidden_size > 0, f"hidden_size must be positive, got {hidden_size}"
    target_layer_ids = [int(layer_id) for layer_id in manifest["target_layer_ids"]]
    assert target_layer_ids, "target_layer_ids must not be empty."  # 必须至少监督一层
    assert target_layer_ids == sorted(target_layer_ids), (  # 升序约定：保证拼接顺序确定性
        "target_layer_ids must be sorted in ascending order."
    )
    num_shards = int(manifest["num_shards"])
    assert num_shards == len(manifest["shards"]), (  # 计数与明细条数一致
        "num_shards does not match shard metadata count: "
        f"{num_shards} != {len(manifest['shards'])}"
    )
    for expected_shard_id, shard in enumerate(manifest["shards"]):
        assert int(shard["shard_id"]) == expected_shard_id, (  # shard_id 必须从 0 连续编号
            "Target cache shard ids must be contiguous starting from 0: "
            f"expected {expected_shard_id}, got {shard['shard_id']}"
        )
        shard_path = build_target_cache_shard_path(cache_dir, shard["file_name"])
        assert os.path.exists(shard_path), f"Missing target cache shard file: {shard_path}"  # 文件实际存在
    index_path = os.path.join(cache_dir, "samples.idx")
    assert os.path.exists(index_path), f"Missing target cache index file: {index_path}"
    index_size = os.path.getsize(index_path)
    assert index_size % INDEX_RECORD_SIZE == 0, (  # 定长记录文件大小必须是记录大小的整数倍
        "samples.idx size is not a multiple of the canonical record size: "
        f"{index_size} % {INDEX_RECORD_SIZE} != 0"
    )
    num_samples = int(manifest["num_samples"])
    assert index_size == num_samples * INDEX_RECORD_SIZE, (  # 索引大小与样本数严格对应（无多余/缺失记录）
        "samples.idx size does not match manifest.num_samples: "
        f"{index_size} != {num_samples} * {INDEX_RECORD_SIZE}"
    )


def validate_train_cache(*, train_dataset, draft_model, target_model_name_or_path):
    # 训练启动前的一致性校验：缓存是为某个特定 draft 结构与 target 模型生成的，
    # 换模型/换层数复用旧缓存会得到形状错配或语义错误的监督信号，必须显式拒绝。
    manifest = train_dataset.manifest
    expected_layer_ids = [int(layer_id) for layer_id in draft_model.target_layer_ids]
    assert [int(layer_id) for layer_id in manifest["target_layer_ids"]] == expected_layer_ids, (  # 监督层逐层对齐
        "Target cache target_layer_ids do not match draft model configuration: "
        f"{manifest['target_layer_ids']} != {expected_layer_ids}"
    )
    assert int(manifest["hidden_size"]) == int(draft_model.config.hidden_size), (  # 隐维度对齐
        "Target cache hidden_size does not match draft model hidden size: "
        f"{manifest['hidden_size']} != {draft_model.config.hidden_size}"
    )
    cache_target_model_name = manifest["target_model_name_or_path"]
    assert str(cache_target_model_name) == str(target_model_name_or_path), (  # target 模型名对齐（防串缓存）
        "Target cache target_model_name_or_path does not match training config: "
        f"{cache_target_model_name} != {target_model_name_or_path}"
    )


def _tensor_to_bytes(tensor: torch.Tensor, dtype: torch.dtype):
    # 张量 → 原生字节流：detach 脱离计算图、搬 CPU、转目标 dtype、保证连续内存后序列化。
    # 任何 GPU 张量都先拷到 CPU，避免在字节层面触发隐式 CUDA 同步。
    cpu_tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return cpu_tensor.numpy().tobytes()


def _tensor_to_bfloat16_bytes(tensor: torch.Tensor):
    # bf16 张量 → 字节流：NumPy 没有 bfloat16 dtype，
    # 技巧是 view(torch.uint16) 把 bf16 按位重解释为 u16（零拷贝、不改比特），
    # 再交 numpy 取原始字节；读取端对称地用 uint16 读入后 view 回 bf16。
    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return cpu_tensor.view(torch.uint16).numpy().tobytes()


def compute_local_sample_range(*, num_samples: int, rank: int, world_size: int):
    # 把 num_samples 尽量均匀切给 world_size 个 rank：前 remainder 个 rank 多摊 1 个。
    # 确定性切分（与 DataLoader 无关），保证导出阶段各 rank 样本区间互不重叠且全覆盖。
    base = int(num_samples) // int(world_size)
    remainder = int(num_samples) % int(world_size)
    start = rank * base + min(rank, remainder)  # 前 remainder 个 rank 各补一个偏移
    local_count = base + (1 if rank < remainder else 0)
    return start, start + local_count  # 左闭右开区间 [start, end)


def prepare_target_cache_output_dir(output_dir: str):
    # 准备导出输出目录：不允许在非空目录上导出（防覆盖已有缓存），并建 _tmp 暂存区。
    # 各 rank 的中间产物先写 _tmp/rank_N，合并完成后再清理。
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        existing = sorted(os.listdir(output_dir))
        if existing:
            raise FileExistsError(  # 空目录才允许作为导出目标
                f"Target cache output dir is not empty: {output_dir}. "
                "Use a new output directory."
            )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "_tmp"), exist_ok=True)  # 各 rank 中间产物目录


@dataclass
class LocalCacheWriteSummary:
    # 单个 rank 导出阶段的产物清单：负责的源样本区间、本地样本/shard 数与文件名列表。
    # rank0 收集各 rank 的 summary 后用于全局合并（改名 shard、拼索引、写 manifest）。
    global_rank: int  # 该 summary 所属的全局 rank
    source_sample_start: int  # 该 rank 负责的源样本区间起点（全局编号）
    source_sample_end: int  # 区间终点（不含）
    num_local_samples: int  # 本地实际写入样本数
    num_local_shards: int  # 本地 shard 文件数
    local_shard_files: list[str]  # 本地 shard 文件名（按本地 shard_id 顺序）

    def to_json(self):
        # 序列化为 summary.json 的内容；列表显式拷贝避免共享可变引用。
        return {
            "global_rank": self.global_rank,
            "source_sample_start": self.source_sample_start,
            "source_sample_end": self.source_sample_end,
            "num_local_samples": self.num_local_samples,
            "num_local_shards": self.num_local_shards,
            "local_shard_files": list(self.local_shard_files),
        }


@dataclass(frozen=True)
class TargetCacheSampleBytes:
    # 一个样本的"字节化"形态：五个张量各自已按协议 dtype 序列化为 bytes。
    # frozen=True：字节串本身不可变，可在异步队列中安全传递、无需防御性拷贝。
    sample_id: int
    seq_len: int
    input_ids: bytes  # int32 小端字节流，长度 = seq_len*4
    attention_mask: bytes  # uint8，长度 = seq_len
    loss_mask: bytes  # uint8，长度 = seq_len
    target_hidden_states: bytes  # bf16（按 u16 存储），长度 = seq_len*L*H*2
    target_last_hidden_states: bytes  # bf16，长度 = seq_len*H*2


def build_target_cache_sample_bytes(
    *,
    sample_id: int,
    input_ids: torch.Tensor,  # [seq_len] int64/long
    attention_mask: torch.Tensor,  # [seq_len]
    loss_mask: torch.Tensor,  # [seq_len]
    target_hidden_states: torch.Tensor,  # [seq_len, L*H] bf16（多中间层展平拼接）
    target_last_hidden_states: torch.Tensor,  # [seq_len, H] bf16
):
    # 把张量样本转为可落盘的字节记录：seq_len 取 input_ids 第一维，各段按协议 dtype 序列化。
    return TargetCacheSampleBytes(
        sample_id=int(sample_id),
        seq_len=int(input_ids.shape[0]),
        input_ids=_tensor_to_bytes(input_ids, torch.int32),  # long → int32 存储省一半空间
        attention_mask=_tensor_to_bytes(attention_mask, torch.uint8),
        loss_mask=_tensor_to_bytes(loss_mask, torch.uint8),
        target_hidden_states=_tensor_to_bfloat16_bytes(target_hidden_states),  # bf16 按 u16 位型存储
        target_last_hidden_states=_tensor_to_bfloat16_bytes(
            target_last_hidden_states
        ),
    )


class LocalTargetCacheWriter:
    # 同步版本地 shard 写入器：把样本字节流追加进当前 shard，满 max_shard_bytes 即滚动开新 shard；
    # 每写一个样本同步追加一条 56 字节索引记录到 samples.local.idx。
    # 单文件上限的意义：几十 GB 级的 shard 才能被文件系统/对象存储/mmap 友好处理。
    def __init__(self, *, rank_dir: str, max_shard_bytes: int):
        self.rank_dir = rank_dir  # 本 rank 的输出目录（通常 output_dir/_tmp/rank_N）
        self.max_shard_bytes = int(max_shard_bytes)  # 单 shard 目标字节上限（软上限：单样本不切分）
        self.local_index_path = os.path.join(rank_dir, "samples.local.idx")  # 本地索引文件
        self.index_handle = open(self.local_index_path, "wb")  # 索引随写随追加
        self.current_shard_id = -1  # 当前 shard 编号（开第一个 shard 时自增为 0）
        self.current_shard_handle = None  # 当前 shard 文件句柄
        self.current_shard_size = 0  # 当前 shard 已写字节数
        self.local_shard_files = []  # 已创建的 shard 文件名（顺序即本地 shard_id）
        self.num_local_samples = 0  # 已写入样本计数

    def close(self):
        # 收尾：fsync 并关闭当前 shard 与索引文件（幂等，二次调用无副作用）。
        # fsync 确保后续合并阶段读到完整数据（合并可能在同进程另一时刻进行）。
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
            self.current_shard_handle = None
        if getattr(self, "index_handle", None) is not None:
            self.index_handle.flush()
            os.fsync(self.index_handle.fileno())
            self.index_handle.close()
            self.index_handle = None

    def _open_new_shard(self):
        # 关闭当前 shard 并开启下一个：文件名 shard-local-XXXXX.bin（本地编号）。
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
        self.current_shard_id += 1
        file_name = f"shard-local-{self.current_shard_id:05d}.bin"  # 定宽编号：字典序即数值序
        shard_path = os.path.join(self.rank_dir, file_name)
        self.current_shard_handle = open(shard_path, "wb")
        self.current_shard_size = 0  # 新 shard 从零计数
        self.local_shard_files.append(file_name)

    def _ensure_shard(self, sample_nbytes: int):
        # shard 容量治理：首个样本开新 shard；之后若再放当前样本会超上限则滚动新 shard。
        # 注意单样本永不切分——即使超过 max_shard_bytes 也独占一个 shard。
        if self.current_shard_handle is None:
            self._open_new_shard()
            return
        if (
            self.current_shard_size > 0
            and self.current_shard_size + int(sample_nbytes) > self.max_shard_bytes
        ):
            self._open_new_shard()

    def write_sample_bytes(self, sample: TargetCacheSampleBytes):
        # 落盘一个字节化样本：五段依次 append 到当前 shard，边写边记录各段偏移，
        # 最后把 (sample_id, shard_id, seq_len, 五偏移) 打包追加进本地索引。
        # 同一 shard 内样本紧凑排列，偏移即"上一段结束处"。
        sample_nbytes = (  # 本样本总字节数（五段之和）
            len(sample.input_ids)
            + len(sample.attention_mask)
            + len(sample.loss_mask)
            + len(sample.target_hidden_states)
            + len(sample.target_last_hidden_states)
        )
        self._ensure_shard(sample_nbytes)
        input_ids_offset = self.current_shard_size  # 写入前记偏移
        self.current_shard_handle.write(sample.input_ids)
        self.current_shard_size += len(sample.input_ids)
        attention_mask_offset = self.current_shard_size
        self.current_shard_handle.write(sample.attention_mask)
        self.current_shard_size += len(sample.attention_mask)
        loss_mask_offset = self.current_shard_size
        self.current_shard_handle.write(sample.loss_mask)
        self.current_shard_size += len(sample.loss_mask)
        target_hidden_states_offset = self.current_shard_size
        self.current_shard_handle.write(sample.target_hidden_states)
        self.current_shard_size += len(sample.target_hidden_states)
        target_last_hidden_states_offset = self.current_shard_size
        self.current_shard_handle.write(sample.target_last_hidden_states)
        self.current_shard_size += len(sample.target_last_hidden_states)
        self.index_handle.write(  # 本地索引记录：与数据同顺序，稍后合并时重编号
            pack_index_record(
                sample_id=sample.sample_id,
                shard_id=self.current_shard_id,
                seq_len=sample.seq_len,
                input_ids_offset=input_ids_offset,
                attention_mask_offset=attention_mask_offset,
                loss_mask_offset=loss_mask_offset,
                target_hidden_states_offset=target_hidden_states_offset,
                target_last_hidden_states_offset=target_last_hidden_states_offset,
            )
        )
        self.num_local_samples += 1

    def write_sample(
        self,
        *,
        sample_id: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
    ):
        # 张量入口：先字节化再走 write_sample_bytes；导出脚本直接调用这个。
        sample = build_target_cache_sample_bytes(
            sample_id=sample_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
        )
        self.write_sample_bytes(sample)


class AsyncTargetCacheWriter:
    # 异步版写入器：把"字节化 + 落盘"挪到后台线程，导出进程（跑目标模型前向的 GPU 侧）
    # 不被磁盘 IO 阻塞。有界队列提供背压：写盘跟不上时生产端阻塞，防止内存膨胀。
    # 关键约束：队列里只放 CPU bytes 记录，绝不持有 CUDA 张量引用（见下）。
    def __init__(
        self,
        *,
        rank_dir: str,
        max_shard_bytes: int,
        max_queue_size: int = 128,
    ):
        self.writer = LocalTargetCacheWriter(  # 实际干活的同步写入器，运行在后台线程
            rank_dir=rank_dir,
            max_shard_bytes=max_shard_bytes,
        )
        # Queue CPU byte records only; never hold CUDA tensor references here.
        self.queue = queue.Queue(maxsize=int(max_queue_size))  # 有界队列：满则 put 阻塞形成背压
        self.sentinel = object()  # 哨兵对象：唯一合法的"关闭"信号（None 可能与坏样本混淆）
        self.num_local_samples = 0  # 生产端计数，用于关闭时与消费端对账
        self._closed = False  # 防止重复 close
        self._exception = None  # 后台线程异常存档：关闭时统一抛出，不让错误静默丢失
        self.thread = threading.Thread(  # 后台落盘线程（非 daemon：close 会 join 它）
            target=self._run,
            name=f"target-cache-writer-{os.path.basename(rank_dir)}",  # 线程名便于日志定位
        )
        self.thread.start()

    @property
    def local_shard_files(self):
        # 透传底层 writer 的 shard 文件名列表（合并阶段需要）。
        return self.writer.local_shard_files

    def _run(self):
        # 消费循环：逐条写盘，直到收到哨兵；任何异常都记录并吞掉（由 close 统一上报），
        # 避免线程裸崩导致主进程死等或丢错误信息。finally 保证无论成败都 close 落盘。
        try:
            while True:
                item = self.queue.get()  # 阻塞取队头
                try:
                    if item is self.sentinel:
                        break  # 收到哨兵：正常退出循环
                    self.writer.write_sample_bytes(item)  # 实际磁盘写
                finally:
                    self.queue.task_done()  # 计数归还（即使抛异常也要标记完成）
        except BaseException as exc:
            self._exception = exc  # 捕获一切异常（含 KeyboardInterrupt），延迟上报
        finally:
            try:
                self.writer.close()  # 无论正常/异常都收尾落盘，已写数据尽量保留
            except BaseException as exc:
                if self._exception is None:
                    self._exception = exc  # close 失败也要记账（优先保留首个异常）

    def _raise_if_failed(self):
        # 后台线程曾失败则立即抛 RuntimeError（保留原始异常链）。
        if self._exception is not None:
            raise RuntimeError("Async target cache writer failed.") from self._exception

    def _put(self, item):
        # 带失败检测的阻塞投递：put 分段超时重试，期间轮询后台线程是否已挂——
        # 否则队列满 + 消费者死亡 = 主线程永久死锁。
        while True:
            self._raise_if_failed()  # 消费者已异常则快速失败
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue  # 队列暂满：1 秒后重试（背压）

    def write_sample(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
    ):
        # 异步写入入口：立即字节化（同步完成 CUDA→CPU 拷贝与 dtype 转换），再投递 bytes。
        # 字节化必须在调用线程做：保证入队对象不含 CUDA 内存，后台线程零 GPU 依赖。
        sample = build_target_cache_sample_bytes(
            sample_id=self.num_local_samples,  # 本地样本号：0 起稠密递增
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
        )
        self._put(sample)
        self.num_local_samples += 1

    def close(self):
        # 关闭流程：投哨兵 → join 线程 → 对账样本数 → 上报异常。
        # 对账（两处计数必须相等）可发现"投递丢失/线程提前死亡"这类静默数据损坏。
        if self._closed:
            self._raise_if_failed()  # 重复 close 仍要暴露历史异常
            return
        if self._exception is None:
            self._put(self.sentinel)  # 仅当后台线程还健康才投哨兵（否则 put 会误等）
        self.thread.join()  # 等消费循环与收尾落盘结束
        self._closed = True
        self._raise_if_failed()  # 线程内异常在此统一抛给调用方
        assert self.writer.num_local_samples == self.num_local_samples, (  # 产出对账
            "Async target cache writer lost samples: "
            f"{self.writer.num_local_samples} != {self.num_local_samples}"
        )


def load_local_cache_write_summary(rank_dir: str):
    # 读取某 rank 的 summary.json（合并阶段 rank0 收集各 rank 产物清单时用）。
    with open(os.path.join(rank_dir, "summary.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_global_target_cache_shard_map(summaries):
    # 全局 shard 编号分配：按 source_sample_start 排序各 rank 的 summary，
    # 依序给每个 rank 的本地 shard 分配连续全局 shard_id（0 起）。
    # 返回 (rank → 本地id→全局id 映射, 全局 shard 元信息列表)。
    # 排序键为样本区间起点，保证最终样本顺序 = 源数据顺序，全局样本号稠密递增。
    shard_map = {}  # {global_rank: [全局 shard_id, ...]}，下标即本地 shard_id
    shards = []  # 全局 shard 元信息 [{"shard_id", "file_name"}, ...]
    next_shard_id = 0  # 全局 shard 计数器
    for summary in sorted(
        summaries,
        key=lambda item: int(item["source_sample_start"]),  # 按样本区间起点排序
    ):
        local_map = []  # 该 rank 的本地→全局 shard 映射
        for _local_shard_id, _file_name in enumerate(summary["local_shard_files"]):
            local_map.append(next_shard_id)
            shards.append(
                {
                    "shard_id": next_shard_id,
                    "file_name": f"shard-{next_shard_id:05d}.bin",  # 全局统一命名
                }
            )
            next_shard_id += 1
        shard_map[int(summary["global_rank"])] = local_map
    return shard_map, shards


def rename_local_target_cache_shards(*, output_dir: str, rank_dir: str, summary, shard_map):
    # 合并第一步：把 rank 目录里的本地 shard 按全局映射改名搬到 output_dir 顶层。
    # os.replace 同盘改名是原子且零拷贝的——几十 TB 数据无需真正复制。
    local_map = shard_map[int(summary["global_rank"])]
    for local_shard_id, file_name in enumerate(summary["local_shard_files"]):
        source = os.path.join(rank_dir, file_name)  # _tmp/rank_N/shard-local-XXXXX.bin
        target = os.path.join(output_dir, f"shard-{local_map[local_shard_id]:05d}.bin")  # 全局名
        os.replace(source, target)


def finalize_target_cache_index(*, output_dir: str, summaries, shard_map):
    # 合并第二步：把各 rank 的本地索引拼成全局 samples.idx。
    # 按 source_sample_start 排序遍历，重编号 sample_id（全局稠密 0 起）并把本地 shard_id
    # 翻译成全局 shard_id；写完 fsync 后原子改名上线。返回全局样本总数。
    index_tmp_path = os.path.join(output_dir, "samples.idx.tmp")
    next_expected_sample_id = 0  # 全局样本号计数器
    with open(index_tmp_path, "wb") as output_handle:
        for summary in sorted(
            summaries,
            key=lambda item: int(item["source_sample_start"]),  # 与 shard 编号分配同一顺序
        ):
            rank_dir = os.path.join(
                output_dir,
                "_tmp",
                f"rank_{int(summary['global_rank'])}",  # 该 rank 的中间产物目录
            )
            local_index_path = os.path.join(rank_dir, "samples.local.idx")
            with open(local_index_path, "rb") as local_handle:
                local_bytes = local_handle.read()  # 本地索引整体读入（56B×样本数，内存可承受）
            assert len(local_bytes) % INDEX_RECORD_SIZE == 0, (  # 定长记录完整性检查
                "Local target cache index has invalid size: "
                f"{local_index_path}"
            )
            next_local_sample_id = 0  # 本地样本号校验计数器
            for offset in range(0, len(local_bytes), INDEX_RECORD_SIZE):
                record = unpack_index_record(local_bytes, offset)
                assert int(record["sample_id"]) == next_local_sample_id, (  # 本地索引必须按 sample_id 有序
                    "Local target cache index is not ordered by local sample_id: "
                    f"got {record['sample_id']}, expected {next_local_sample_id}"
                )
                record["sample_id"] = next_expected_sample_id  # 重编号为全局连续 id
                record["shard_id"] = shard_map[int(summary["global_rank"])][  # 本地→全局 shard 翻译
                    int(record["shard_id"])
                ]
                output_handle.write(pack_index_record(**record))  # 重新打包写入全局索引
                next_local_sample_id += 1
                next_expected_sample_id += 1
        output_handle.flush()
        os.fsync(output_handle.fileno())  # 强制落盘
    os.replace(index_tmp_path, os.path.join(output_dir, "samples.idx"))  # 原子上线
    return next_expected_sample_id  # 即全局 num_samples


def build_target_cache_manifest(
    *,
    num_samples: int,
    shards,
    target_layer_ids,
    hidden_size: int,
    extra_fields=None,
):
    # 构造 manifest：协议常量（版本/dtype/记录大小）+ 运行时事实（样本数/shard/监督层/隐维度）。
    # extra_fields 用于附加溯源信息（如 target_model_name_or_path）。
    manifest = {
        "version": TARGET_CACHE_VERSION,
        "num_samples": int(num_samples),
        "num_shards": len(shards),
        "target_layer_ids": [int(layer_id) for layer_id in target_layer_ids],
        "hidden_dtype": TARGET_CACHE_HIDDEN_DTYPE,
        "token_dtype": TARGET_CACHE_TOKEN_DTYPE,
        "mask_dtype": TARGET_CACHE_MASK_DTYPE,
        "index_record_size": INDEX_RECORD_SIZE,
        "hidden_size": int(hidden_size),
        "shards": shards,
    }
    if extra_fields:
        manifest.update(extra_fields)
    return manifest


def write_target_cache_manifest(*, output_dir: str, manifest):
    # manifest 最后写入：它是缓存"生效"的开关——存在且合法即整份缓存可用。
    # 原子写保证崩溃时不留半份 manifest 误导读取端。
    atomic_json_dump(manifest, os.path.join(output_dir, "manifest.json"))


def cleanup_target_cache_tmp_dir(output_dir: str):
    # 删除 _tmp 目录：合并完成后各 rank 中间产物（本地索引）不再需要。
    # 注意：本地 shard 已被 rename 走，_tmp 里只剩索引文件。
    tmp_dir = os.path.join(output_dir, "_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


class CacheDataset(torch.utils.data.Dataset):
    # 目标缓存读取端：训练时的 Dataset。三层数据组织：
    #   manifest.json（元信息） → samples.idx（56B/样本定长索引，mmap） → shard-*.bin（样本字节）。
    # __getitem__ 全程 mmap + np.frombuffer 零拷贝视图（随即 copy 出独立数组），
    # 多进程 DataLoader worker 各自映射同一文件页缓存，随机读几十 TB 缓存也只按需缺页。
    def __init__(self, cache_dir: str, max_open_shards: int = 4):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)  # 缓存根目录（绝对路径）
        self.manifest = load_target_cache_manifest(self.cache_dir)  # 读取并校验 manifest
        self.num_samples = int(self.manifest["num_samples"])  # 全局样本总数
        self.hidden_size = int(self.manifest["hidden_size"])  # 隐状态维度
        self.target_layer_ids = [int(layer_id) for layer_id in self.manifest["target_layer_ids"]]  # 监督层列表
        self.num_target_layers = len(self.target_layer_ids)  # 层数 L：决定 hidden_states 的第二维
        self.index_path = os.path.join(self.cache_dir, "samples.idx")  # 定长索引文件
        self.index_file = None  # 索引文件句柄（惰性打开）
        self.index_mmap = None  # 索引 mmap（惰性建立）
        self.max_open_shards = max_open_shards  # 同时打开的 shard 上限：LRU 控制句柄/映射数量
        self.shard_handles = OrderedDict()  # shard_id → 文件句柄（OrderedDict 维护 LRU 顺序）
        self.shard_mmaps = OrderedDict()  # shard_id → mmap
        self.shard_paths = {  # shard_id → 文件绝对路径（来自 manifest，只读一次）
            int(shard["shard_id"]): build_target_cache_shard_path(
                self.cache_dir,
                shard["file_name"],
            )
            for shard in self.manifest["shards"]
        }

    def __len__(self):
        # 样本总数，供 sampler 切分与 epoch 规划。
        return self.num_samples

    def close(self):
        # 释放全部 mmap 与句柄（幂等）；getattr 防御：__del__ 可能在部分初始化后触发。
        for shard_mmap in getattr(self, "shard_mmaps", {}).values():
            shard_mmap.close()
        for handle in getattr(self, "shard_handles", {}).values():
            handle.close()
        if hasattr(self, "shard_mmaps"):
            self.shard_mmaps.clear()
        if hasattr(self, "shard_handles"):
            self.shard_handles.clear()
        if getattr(self, "index_mmap", None) is not None:
            self.index_mmap.close()
            self.index_mmap = None
        if getattr(self, "index_file", None) is not None:
            self.index_file.close()
            self.index_file = None

    def __del__(self):  # pragma: no cover
        # 析构兜底：确保映射与 fd 释放，避免 DataLoader worker 退出时泄漏句柄。
        self.close()

    def __getstate__(self):  # pragma: no cover
        # 多进程 pickle 定制：剥离不可跨进程传递的句柄/mmap 状态，
        # worker 各自通过惰性 _ensure_index_mmap / _get_shard_mmap 重建。
        state = dict(self.__dict__)
        state["index_file"] = None
        state["index_mmap"] = None
        state["shard_handles"] = OrderedDict()
        state["shard_mmaps"] = OrderedDict()
        return state

    def _ensure_index_mmap(self):
        # 惰性建立索引 mmap：首次取样本时才打开 samples.idx。
        if self.index_mmap is None:
            self.index_file = open(self.index_path, "rb")
            self.index_mmap = mmap.mmap(  # 只读映射整个索引（num_samples×56B）
                self.index_file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )

    def _get_shard_mmap(self, shard_id: int):
        # 获取 shard mmap：命中则提到 LRU 队尾；未命中则打开并建立映射，
        # 超过 max_open_shards 时淘汰最久未用的 shard（popitem(last=False) 取队头）。
        # 限制同时映射数：几十 TB 分成数百个 shard，全开会耗尽 fd 与虚拟地址空间。
        shard_id = int(shard_id)
        if shard_id in self.shard_mmaps:
            self.shard_mmaps.move_to_end(shard_id)  # 命中：移到队尾（最近使用）
            self.shard_handles.move_to_end(shard_id)
            return self.shard_mmaps[shard_id]
        shard_path = self.shard_paths[shard_id]
        handle = open(shard_path, "rb")
        self.shard_handles[shard_id] = handle
        self.shard_mmaps[shard_id] = mmap.mmap(  # 只读映射整个 shard
            handle.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )
        while len(self.shard_mmaps) > self.max_open_shards:
            evicted_id, evicted_mmap = self.shard_mmaps.popitem(last=False)  # LRU 淘汰队头
            evicted_mmap.close()
            self.shard_handles.pop(evicted_id).close()
        return self.shard_mmaps[shard_id]

    def _read_record(self, index: int):
        # 读取第 index 条索引记录：偏移 = index × 56（定长记录的 O(1) 寻址）。
        # 同时校验 sample_id == index：保证索引按全局样本号稠密有序，
        # 若不符说明索引损坏或被乱序合并，立即失败。
        self._ensure_index_mmap()
        offset = int(index) * INDEX_RECORD_SIZE
        record = unpack_index_record(self.index_mmap, offset)
        assert int(record["sample_id"]) == int(index), (
            "Target cache index is not sorted by sample_id or sample ids are not dense: "
            f"record sample_id={record['sample_id']}, expected {index}"
        )
        return record

    def _read_tensor_from_shard(
        self,
        *,
        shard_mmap,  # 所在 shard 的 mmap
        offset: int,  # 该段在 shard 内的字节偏移
        shape,  # 目标张量形状（标量维度可广播成 tuple）
        np_dtype,  # NumPy 侧 dtype（与磁盘布局一致）
        torch_dtype,  # torch 侧 dtype（与 np_dtype 应一致，用于校验）
        nbytes: int,  # 该段字节长度，用于越界校验
    ):
        # 通用段读取：frombuffer 直接在 mmap 缓冲上建视图（零拷贝），.copy() 脱离 mmap
        # 生成独立内存（视图会钉住整页且生命周期纠缠，copy 后 mmap 可被安全淘汰）。
        assert int(offset) + int(nbytes) <= shard_mmap.size(), (
            "Target cache tensor extends beyond shard size: "
            f"offset={offset}, nbytes={nbytes}, shard_size={shard_mmap.size()}"
        )
        array = np.frombuffer(
            shard_mmap,
            dtype=np_dtype,
            count=int(np.prod(shape)),  # 元素总数 = 形状各维乘积
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array).view(*shape)  # 共享内存的 torch 视图并整形
        if tensor.dtype != torch_dtype:
            tensor = tensor.to(dtype=torch_dtype)  # dtype 兜底转换（正常路径应已一致）
        return tensor

    def _read_bfloat16_tensor_from_shard(
        self,
        *,
        shard_mmap,
        offset: int,
        shape,
        nbytes: int,
    ):
        # bf16 段读取：磁盘上按 uint16 位型存储（NumPy 无 bf16），
        # 读入 uint16 数组后 view(torch.bfloat16) 按位重解释，无任何数值转换开销。
        assert int(offset) + int(nbytes) <= shard_mmap.size(), (
            "Target cache tensor extends beyond shard size: "
            f"offset={offset}, nbytes={nbytes}, shard_size={shard_mmap.size()}"
        )
        array = np.frombuffer(
            shard_mmap,
            dtype=np.uint16,  # bf16 的位型等价物
            count=int(np.prod(shape)),
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array).view(torch.bfloat16)  # 位重解释：u16 → bf16
        return tensor.view(*shape)

    def __getitem__(self, index: int):
        # 单样本读取主流程：查索引 → 取 shard mmap → 按记录内偏移读五段 → 组 dict 返回。
        # 注意不返回 attention_mask：训练端由 CacheCollator 从 padding 布局重建。
        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        record = self._read_record(int(index))  # 56B 索引记录
        seq_len = int(record["seq_len"])
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        shard_mmap = self._get_shard_mmap(int(record["shard_id"]))  # LRU 取映射
        nbytes = expected_target_cache_tensor_nbytes(  # 各段期望字节数（用于越界校验）
            seq_len=seq_len,
            hidden_size=self.hidden_size,
            num_target_layers=self.num_target_layers,
        )
        input_ids = self._read_tensor_from_shard(  # [seq_len] int32
            shard_mmap=shard_mmap,
            offset=record["input_ids_offset"],
            shape=(seq_len,),
            np_dtype=np.int32,
            torch_dtype=torch.int32,
            nbytes=nbytes["input_ids"],
        )
        loss_mask = self._read_tensor_from_shard(  # [seq_len] uint8：assistant 位掩码
            shard_mmap=shard_mmap,
            offset=record["loss_mask_offset"],
            shape=(seq_len,),
            np_dtype=np.uint8,
            torch_dtype=torch.uint8,
            nbytes=nbytes["loss_mask"],
        )
        target_hidden_states = self._read_bfloat16_tensor_from_shard(  # [seq_len, L*H] 多层展平
            shard_mmap=shard_mmap,
            offset=record["target_hidden_states_offset"],
            shape=(seq_len, self.num_target_layers * self.hidden_size),
            nbytes=nbytes["target_hidden_states"],
        )
        target_last_hidden_states = self._read_bfloat16_tensor_from_shard(  # [seq_len, H] 末层
            shard_mmap=shard_mmap,
            offset=record["target_last_hidden_states_offset"],
            shape=(seq_len, self.hidden_size),
            nbytes=nbytes["target_last_hidden_states"],
        )
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden_states,
            "target_last_hidden_states": target_last_hidden_states,
        }


def _pad_1d_batch(features: List[Dict], key: str):
    # 1D 张量按批内最长序列右侧补零成 [B, max_length]；零填充的 token 由
    # attention_mask/loss_mask 在损失端排除，不影响训练语义。
    max_length = max(item[key].shape[0] for item in features)  # 批内最长（变长序列的 padding 基准）
    batch_size = len(features)
    dtype = features[0][key].dtype  # 与样本同 dtype（int32/uint8 等）
    out = torch.zeros((batch_size, max_length), dtype=dtype)  # 预填零：padding 区天然为 0
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]  # 每行前缀放入真实内容
    return out


def _pad_hidden_batch(features: List[Dict], key: str):
    # 2D 隐状态按批内最长序列补零成 [B, max_length, hidden_dim]；零向量不产生有效监督。
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    hidden_dim = features[0][key].shape[1]  # hidden 维度（L*H 或 H）
    dtype = features[0][key].dtype  # 通常为 bfloat16
    out = torch.zeros((batch_size, max_length, hidden_dim), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


class ConversationCollator:
    # 原始对话数据的整理器（未走目标缓存时使用）：逐样本 tokenize + loss_mask，
    # 过滤掉损失 token 不足 min_loss_tokens 的"无效样本"（如回复过短/纯截断样本）。
    def __init__(
        self,
        tokenizer,
        chat_template,
        max_length,
        min_loss_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template  # 注册表中的模板名（字符串）
        self.max_length = int(max_length)
        self.min_loss_tokens = int(min_loss_tokens)  # 损失 token 数下限：低于此值的样本直接丢弃

    def _process_feature(self, item):
        # 单样本预处理；损失过少返回 None（由 __call__ 统一过滤）。
        processed = preprocess_record(
            record=item,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            max_length=self.max_length,
        )
        if int(processed["loss_mask"].sum().item()) < self.min_loss_tokens:
            return None  # assistant 部分太短：训练价值低，跳过
        return processed

    def __call__(self, features: List[Dict]):
        # DataLoader collate 接口：处理 → 剔除无效样本 → 按 1D 键补零成批。
        # 全部样本无效时返回 None（训练侧需处理空批）。
        features = [self._process_feature(item) for item in features]
        features = [item for item in features if item is not None]
        if not features:
            return None
        batch = {}
        for key in ("input_ids", "attention_mask", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)
        return batch


class CacheCollator:
    # 目标缓存样本的整理器：样本已是定长段，只需按批内最长序列补零。
    # attention_mask 特殊处理：从 padding 布局重建（前 seq_len 位为 1），
    # 因为 CacheDataset 为了省存储没有把全 1 的 attention_mask 写进 shard。
    def __call__(self, features: List[Dict]):
        batch = {}
        for key in ("input_ids", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)  # 1D 键补零
        attention_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)  # 全 0 起步
        for i, item in enumerate(features):
            attention_mask[i, : item["input_ids"].shape[0]] = 1  # 真实 token 区间置 1
        batch["attention_mask"] = attention_mask
        for key in ("target_hidden_states", "target_last_hidden_states"):
            batch[key] = _pad_hidden_batch(features, key)  # 2D 隐状态补零
        return batch
