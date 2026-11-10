# 文件系统小工具：递归建目录与"原子"符号链接更新。
import os  # makedirs 与 os.replace（原子替换系统调用）
from pathlib import Path  # 路径对象与 symlink 操作


# 递归创建目录：已存在则静默跳过（exist_ok），多进程并发创建安全。
def ensure_dir(path):  # path：目录路径，可多级不存在
    os.makedirs(path, exist_ok=True)  # 已存在不抛错


# 原子更新符号链接：先写临时链接 .tmp，再 os.replace 原子换名，读方永远能看到
# 完整有效的 dst，不会出现链接短暂消失或指向旧目标的中间窗口（用于 latest 链接等）。
def safe_symlink(src, dst):  # src：链接指向的目标；dst：符号链接路径
    # os.replace is atomic, so dst never disappears mid-update.
    dst_path = Path(dst)  # 目标链接路径
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")  # 同目录临时名（保证与 dst 同文件系统，replace 才原子）
    if tmp_path.is_symlink() or tmp_path.exists():  # 清理上次异常残留的临时链接
        tmp_path.unlink()
    tmp_path.symlink_to(Path(src).resolve())  # 建临时链接；src 转绝对路径，避免链接被移动后失效
    os.replace(tmp_path, dst_path)  # 原子替换：rename 系统调用，dst 无中间态
