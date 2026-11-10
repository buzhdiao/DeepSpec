# 常量子模块的入口：按运行环境二选一导出常量集。
# 在 hfai（调度平台）环境优先用 internal 版（可能含平台相关路径），
# 普通环境回退到 public 版通用常量。
try:
    import hfai  # 探测 hfai 平台客户端是否可用
    from .internal import *  # 平台环境：使用内部常量集
except:
    from .public import *  # 普通环境：使用公共常量集（裸 except 兼容各类导入失败）