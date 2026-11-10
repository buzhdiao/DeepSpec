# hfai（调度平台）挂起恢复机制：后台线程轮询平台的挂起信号，训练循环在检查点处
# 通过 requested() 同步全队意志，一致决定是否立即存盘并 go_suspend 让出机器。
import contextlib  # contextmanager 装饰器
import threading  # 后台轮询线程与事件

import torch  # 跨 rank 广播用的标志张量
import torch.distributed as dist  # barrier/broadcast 集合通信

from .distributed import is_global_main_process, print_on_global_main  # 主进程判断与带时间戳打印

try:
    import hfai  # 平台客户端：仅在调度环境中可用

    HAS_HFAI = True  # 环境支持 hfai 挂起
except ModuleNotFoundError:
    hfai = None  # 占位，避免引用未定义名字
    HAS_HFAI = False  # 本地/普通环境：挂起相关逻辑全部空转


# 挂起控制器：平台信号是节点级的，监测线程只在全局主进程上跑；requested() 用
# barrier + broadcast 把主进程收到的信号同步给所有 rank，保证全队在同一时刻做出
# 一致的挂起决定，避免部分 rank 存盘、部分 rank 继续训练导致状态分叉。
class SuspendController:
    def __init__(self, device, poll_interval_seconds: float = 1.0):  # device：标志张量所在设备；poll_interval_seconds：轮询间隔（秒）
        self.device = device  # 保存设备，用于创建跨 rank 广播的张量
        self.poll_interval_seconds = float(poll_interval_seconds)  # 轮询周期：越小响应越快、开销略增
        self._requested = threading.Event()  # 主进程收到挂起信号后置位
        self._stop_requested = threading.Event()  # 通知监测线程退出
        self._monitor_thread = None  # 监测线程句柄（仅主进程创建）
        self._suspend_flag = torch.zeros(1, device=self.device, dtype=torch.int32)  # 广播缓冲：0/1 表示是否挂起

    # 监测线程主循环：每隔 poll_interval_seconds 非阻塞查询一次平台挂起命令；
    # 收到后置位 _requested 并退出线程，训练循环在下一检查点通过 requested() 感知。
    def _monitor_suspend(self):
        print_on_global_main("Start monitoring hfai suspend signal...")  # 日志：监测已启动
        while not self._stop_requested.is_set():  # 未被要求停止就一直轮询
            if hfai.client.receive_suspend_command():  # 非阻塞查询平台是否下发挂起命令
                print_on_global_main("Received hfai suspend command!")  # 日志：收到挂起指令
                self._requested.set()  # 记录挂起请求，供 requested() 读取
                return  # 信号收一次即可，线程退出
            self._stop_requested.wait(timeout=self.poll_interval_seconds)  # 用可中断等待实现轮询间隔

    # 上下文管理器：进入时清理旧状态并在全局主进程上拉起监测线程；退出时（无论
    # 正常结束还是抛异常）先通知线程退出再 join，防止线程泄漏。无 hfai 环境原样放行。
    @contextlib.contextmanager
    def monitoring(self):
        if not HAS_HFAI:  # 无平台环境
            yield self  # 空转：调用方代码路径不变
            return

        try:  # 保证退出时必清理线程
            self._requested.clear()  # 清掉上次可能残留的挂起请求
            self._stop_requested.clear()  # 重置线程停止标志
            if is_global_main_process():  # 信号是节点级的，只需主进程监听
                self._monitor_thread = threading.Thread(  # 创建监测线程
                    target=self._monitor_suspend,  # 线程主循环
                    daemon=True,  # daemon：主进程退出时不被线程卡住
                )
                self._monitor_thread.start()  # 启动轮询
            yield self  # 训练循环在 with 块内运行，期间可随时调 requested()
        finally:  # 退出 with 块后必定执行
            self._stop_requested.set()  # 让轮询线程尽快退出
            if self._monitor_thread is not None:  # 仅主进程有线程可收
                self._monitor_thread.join(timeout=self.poll_interval_seconds + 1.0)  # 等线程退出（最多一个轮询周期+1s）
                self._monitor_thread = None  # 释放句柄

    # 训练循环在检查点处询问"是否挂起"：先 barrier 对齐，再由主进程把挂起标志
    # 广播给全体，所有 rank 拿到同一答案，硬件/轮询时序差异不会造成分叉。
    def requested(self) -> bool:
        if not HAS_HFAI:  # 无平台环境永远不挂起
            return False

        dist.barrier()  # 全队先在此对齐，再进入广播
        if is_global_main_process():  # 只有主进程能读到监测线程的结果
            self._suspend_flag[0] = 1 if self._requested.is_set() else 0  # 把 Python 布尔写入标志张量
        dist.broadcast(self._suspend_flag, src=0)  # 广播给所有 rank（集合通信，全体必须同时调用）
        return bool(self._suspend_flag.item())  # 各 rank 返回统一的挂起决定

    # 真正执行挂起：通知平台本作业已完成存盘、让出 GPU；调用前必须先保存 checkpoint，
    # 平台恢复调度后从续训点继续。
    def go_suspend(self):
        if not HAS_HFAI:  # 非平台环境调用视为程序错误
            raise RuntimeError("hfai is not available; cannot suspend the job.")
        hfai.client.go_suspend()  # 平台调用：作业进入挂起队列
