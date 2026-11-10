# 学习率调度与 bf16 混合精度优化器：提供"warmup + 任意后续调度"的两段式调度器，
# 以及保持 fp32 主参数副本的 AdamW 封装（模型权重以 bf16 参与前后向，优化在 fp32 上）。
import torch  # 张量运算与 AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR  # 余弦退火（作两段式的第二段）
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler  # 调度器基类


# 两段式调度器：前段由本类控制（如线性 warmup），越过切换点后把学习率控制权
# 移交 after_scheduler；state_dict 对内嵌调度器做了"类型名 + 状态"拆分序列化。
class TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):  # optimizer：绑定优化器；after_scheduler：第二段调度器；last_epoch：起始步（-1 从头）
        self.after_scheduler = after_scheduler  # 第二段调度器（如余弦退火）
        self.finished = False  # 是否已切换到第二段
        super().__init__(optimizer, last_epoch)  # 基类初始化会先调一次 step() 设定初始 lr

    # 序列化：本类自有属性 + 把 after_scheduler 拆成"类型名 + 自身状态"，
    # 避免 pickle 整个调度器对象，断点存档更小、跨版本更稳。
    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key != "optimizer"  # optimizer 不入档（可由模型重建）
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):  # 目前只支持内嵌 LRScheduler 实例
            state_dict["after_scheduler_type"] = type(  # 记录第二段调度器的类名
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[  # 记录其内部状态（last_epoch 等）
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]  # 原对象不再入档
        else:
            raise NotImplementedError()  # 其他形态（函数/lambda）暂不支持
        return state_dict

    # 反序列化：先恢复第二段调度器状态，再把剩余字段交回基类；键集合与 state_dict 对称。
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])  # 恢复第二段内部状态
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")  # 这两键属于第二段，已消费
        }
        super().load_state_dict(state_dict)  # 恢复基类字段（base_lrs、last_epoch 等）


# warmup 调度器：前 warmup_epochs 步按 (step+1)/warmup 线性升到 base_lr，
# 之后整体移交 after_scheduler（首步把 base_lrs 同步给它，使其从满 lr 开始退火）。
class WarmupScheduler(TwoStageScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):  # warmup_epochs：线性预热步数
        self.warmup_epochs = int(warmup_epochs)  # 预热长度
        super().__init__(optimizer, after_scheduler, last_epoch)

    # 计算当前步学习率：预热期线性上升；过预热点后交给第二段
    # （base_lrs 只需在切换后的首步同步一次）。
    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:  # 已过预热期
            if not self.finished:  # 切换后的第一步
                self.after_scheduler.base_lrs = self.base_lrs  # 把初始 lr 交给第二段作退火起点
                self.finished = True  # 标记同步完成
            return self.after_scheduler.get_lr()  # 之后完全由第二段决定

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]  # 线性 warmup：第 k 步 lr = k/warmup × base_lr

    # step 转发：切换后直接驱动第二段并回填 _last_lr（指定 epoch 时要扣掉预热长度，
    # 对齐第二段的步数坐标）；预热期沿用基类逻辑（内部调 get_lr）。
    def step(self, epoch=None):
        if self.finished:  # 第二段阶段
            if epoch is None:  # 常规逐步推进
                self.after_scheduler.step(None)  # 第二段自增一步
                self._last_lr = self.after_scheduler.get_last_lr()  # 同步对外可见的 _last_lr
            else:  # 指定 epoch 推进（断点恢复场景）
                self.after_scheduler.step(epoch - self.warmup_epochs)  # 扣除预热段
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)  # 预热期走基类 step


# "线性 warmup + 余弦退火"一步到位：余弦段 T_max 设为总步数减预热步数，
# 保证在 total_steps 处恰好退火到 eta_min。
class CosineAnnealingWarmupLR(WarmupScheduler):
    def __init__(
        self,
        optimizer,  # 绑定的优化器
        total_steps: int,  # 总训练步数（含预热）
        warmup_steps: int = 0,  # 预热步数
        eta_min: float = 0.0,  # 余弦退火终点学习率
        last_epoch: int = -1,  # 起始步（-1 从头开始）
    ):
        base_scheduler = _CosineAnnealingLR(  # 第二段：余弦退火
            optimizer,
            total_steps - warmup_steps,  # 余弦段长度 = 总步数 - 预热步数
            eta_min=eta_min,  # 退火下界
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)  # 组成两段式


# bf16 混合精度优化器：模型权重保持 bf16 参与前后向（省显存、提速），
# 同时为每个可训练参数维护一份 fp32 主参数，所有 AdamW 更新与权重衰减都发生在
# fp32 上，避免 bf16 直接累加小更新的精度损失；每步结束把主参数写回 bf16 权重。
class BF16Optimizer:
    # Adapted from SpecForge/specforge/optimizer.py:BF16Optimizer.
    def __init__(
        self,
        model,  # 被优化的模型
        lr,  # 峰值学习率
        total_steps,  # 总训练步数（供调度器）
        warmup_ratio,  # 预热步数占比（0~1）
        weight_decay=0.0,  # AdamW 权重衰减系数
    ):
        self.model = model  # 保存模型引用
        self.model_params = [p for p in model.parameters() if p.requires_grad]  # 仅收集参与训练的参数
        self.fp32_params = [  # fp32 主参数：真正的优化对象
            p.detach().clone().to(torch.float32) for p in self.model_params  # 与模型参数一一对应的独立副本
        ]
        for param in self.fp32_params:  # 副本需接收梯度
            param.requires_grad = True
        self.optimizer = torch.optim.AdamW(  # AdamW 作用在 fp32 主参数上
            self.fp32_params, lr=lr, weight_decay=weight_decay  # 解耦权重衰减的 AdamW
        )
        self.scheduler = CosineAnnealingWarmupLR(  # 线性 warmup + 余弦退火
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),  # 占比换算成步数
        )

    # 一步优化：bf16 梯度升精度挂到 fp32 主参数 → AdamW 更新主参数 → 调度器推进 →
    # 主参数写回 bf16 模型权重并释放梯度引用。
    def step(self):
        with torch.no_grad():  # 梯度搬运不需要计算图
            for model_param, master_param in zip(self.model_params, self.fp32_params):  # 逐参数配对
                master_param.grad = (  # bf16 梯度转 fp32 挂到主参数上
                    model_param.grad.detach().to(torch.float32)
                    if model_param.grad is not None  # 未参算的参数无梯度则置 None
                    else None
                )
        self.optimizer.step()  # AdamW 在 fp32 上更新（精度无损的累加）
        self.optimizer.zero_grad()  # 清主参数梯度
        self.scheduler.step()  # 推进学习率调度
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):  # 逐参数写回
                model_param.data.copy_(master_param.data.to(model_param.dtype))  # fp32 → bf16 写回模型权重
                model_param.grad = None  # 释放模型参数上的梯度引用（防泄漏）

    # 序列化：AdamW/调度器状态 + fp32 主参数（搬 CPU 存储），三件齐全即可完整恢复训练。
    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),  # AdamW 动量等状态
            "scheduler_state_dict": self.scheduler.state_dict(),  # 调度器进度
            "fp32_params": [param.detach().cpu() for param in self.fp32_params],  # fp32 主参数本体
        }

    # 恢复：先灌 AdamW/调度器状态，再回填 fp32 主参数，最后同步到 bf16 模型权重。
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])  # 恢复优化器状态
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])  # 恢复调度器进度
        fp32_params = state_dict["fp32_params"]  # 存档中的主参数
        for dst, src in zip(self.fp32_params, fp32_params):  # 逐参数回填
            dst.data.copy_(src.to(dst.device))  # 搬回原设备
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):  # 逐参数同步
                model_param.data.copy_(master_param.data.to(model_param.dtype))  # 使 bf16 权重与主参数一致

    # 返回当前学习率（所有参数组同 lr，取第一组即可）。
    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]  # 第一参数组的当前 lr
