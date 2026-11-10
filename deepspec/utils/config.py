# 配置加载与覆盖机制：把 config/*.py 当独立模块执行并收集顶层变量为 ConfigNode，
# 支持属性式访问、--opts 点路径覆盖（YAML 解析标量）、finalize_cfg 后处理钩子，
# 以及把配置快照安全 dump 成 JSON。
import importlib.util  # 把配置文件动态加载为 Python 模块
import json  # JSON 编码（CustomJSONEncoder 继承其 JSONEncoder）
import os  # os.path.abspath：路径绝对化
from argparse import Namespace  # 兼容 argparse 配置对象（序列化时转字典）
from pathlib import Path  # 路径对象（序列化时转 str）
from types import FunctionType, ModuleType, SimpleNamespace  # 函数/模块/命名空间类型判断
import torch  # 识别 torch.dtype 以便 JSON 序列化
import yaml  # 用 YAML 解析 --opts 传入的标量值（"0.001"→float、"true"→bool 等）


# 继承 dict 的配置节点：支持 cfg.model.block_size 属性式访问，同时保留 dict 全部能力；
# 属性与字典键始终同步。取不存在的键抛 AttributeError（而非 KeyError），
# 与普通对象属性语义一致，hasattr 等内省才能正常工作。
class ConfigNode(dict):  # 以 dict 为存储后端
    def __getattr__(self, item):  # 仅在常规属性查找失败时被调用
        try:
            return self[item]  # 转为字典取值
        except KeyError as exc:  # 键不存在
            raise AttributeError(item) from exc  # 伪装成属性错误，保留原始上下文

    def __setattr__(self, key, value):  # 属性赋值直接写入字典
        self[key] = value

    def copy(self):  # 浅拷贝并保持 ConfigNode 类型（dict.copy 会退化为普通 dict）
        return ConfigNode(super().copy())


# 递归把 dict/list/tuple 转成 ConfigNode 树，使任意层级配置都能属性访问；
# ConfigNode 自身也会逐键重建为全新对象，非容器标量原样返回。
def to_config_node(value):  # value：任意配置值
    if isinstance(value, ConfigNode):  # 已是 ConfigNode：逐键深拷贝重建
        return ConfigNode({k: to_config_node(v) for k, v in value.items()})
    if isinstance(value, dict):  # 普通 dict 转 ConfigNode
        return ConfigNode({k: to_config_node(v) for k, v in value.items()})
    if isinstance(value, list):  # 列表保持 list 类型，仅递归转换元素
        return [to_config_node(v) for v in value]
    if isinstance(value, tuple):  # 元组同理递归
        return tuple(to_config_node(v) for v in value)
    return value  # 标量（int/str/float/bool/None 等）原样返回


# to_config_node 的逆变换：ConfigNode → 普通 dict，供 JSON 序列化等纯数据场景使用；
# 注意 tuple 在此被转成 list（JSON 没有元组）。
def config_to_plain_dict(value):  # value：任意配置值
    if isinstance(value, ConfigNode):  # 配置节点转普通字典
        return {k: config_to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, dict):  # 普通 dict 同样递归
        return {k: config_to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):  # 列表递归
        return [config_to_plain_dict(v) for v in value]
    if isinstance(value, tuple):  # 元组转为列表（JSON 无元组）
        return [config_to_plain_dict(v) for v in value]
    return value  # 标量原样返回


# 递归把值转换为 json.dumps 可直接处理的结构：Path→str、tuple→list、容器逐层下钻；
# 函数/dtype 等特殊对象留给 CustomJSONEncoder 处理。
def jsonable(value):  # value：任意配置值
    if isinstance(value, Path):  # 路径对象转字符串
        return str(value)
    if isinstance(value, ConfigNode):  # 配置节点转普通字典
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):  # 元组转列表
        return [jsonable(item) for item in value]
    if isinstance(value, list):  # 列表逐元素递归
        return [jsonable(item) for item in value]
    if isinstance(value, dict):  # 字典逐键递归
        return {key: jsonable(item) for key, item in value.items()}
    return value  # 标量原样返回


# 自定义 JSON 编码器：把配置中不可直接序列化的对象转成可读字符串/字典，
# 保证把配置快照写入训练日志时不因个别字段报 TypeError 而中断。
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):  # json 遇到未知类型时的回调
        if isinstance(obj, FunctionType):  # 函数（如 trainer_cls）：只记录名字
            return f"<function {obj.__name__}>"
        if isinstance(obj, type):  # 类对象：记录类名
            return f"<class '{obj.__name__}'>"
        if isinstance(obj, torch.dtype):  # torch 数据类型：转字符串如 "torch.bfloat16"
            return str(obj)
        if isinstance(obj, Path):  # 路径转字符串
            return str(obj)
        if isinstance(obj, ConfigNode):  # 配置节点转普通字典
            return config_to_plain_dict(obj)
        if isinstance(obj, (Namespace, SimpleNamespace)):  # argparse/命名空间对象转字典
            return vars(obj)  # 取 __dict__
        return super().default(obj)  # 其余类型交回默认实现（抛 TypeError）


# 用 YAML 解析 --opts 传入的标量字符串："0.001"→float、"true"→True、"None"→None，
# 避免所有覆盖值都退化成 str 导致训练行为意外。
def _parse_scalar(value):  # value：命令行上的原始字符串
    return yaml.safe_load(value)  # safe_load：只解析基本类型，不执行任意构造器


# 核心加载逻辑：把配置 .py 当独立模块执行，收集所有非下划线开头的顶层变量作为
# 配置项（import 进来的模块本身除外），最终包成 ConfigNode 树返回。
# 优势：配置即代码，可用表达式、引用常量；文件里 import 的工具不会混进配置。
def load_config(path):  # path：配置文件路径（config/*.py）
    path = os.path.abspath(path)  # 转绝对路径，避免相对路径受工作目录影响
    module_name = Path(path).stem  # 取文件名（去扩展名）作模块名
    spec = importlib.util.spec_from_file_location(module_name, path)  # 构造按文件路径加载的模块规格
    module = importlib.util.module_from_spec(spec)  # 创建空模块对象
    spec.loader.exec_module(module)  # 执行配置文件，顶层变量挂到模块命名空间
    config = {}  # 收集配置项
    for name in dir(module):  # 遍历模块全部属性
        if name.startswith("__"):  # 跳过 __name__/__doc__ 等内置属性
            continue
        value = getattr(module, name)  # 取变量值
        if isinstance(value, ModuleType):  # 跳过 import 进来的模块本身
            continue
        config[name] = value  # 其余顶层变量视为配置项
    return to_config_node(config)  # 包成 ConfigNode 树


# 应用配置文件中定义的 finalize_cfg 钩子（若存在）：钩子对整体配置做最后加工
# （如拼接 checkpoint/TensorBoard 输出目录），返回 None 则沿用原配置。
def finalize_config(cfg):  # cfg：完整配置
    cfg = to_config_node(cfg)  # 先统一转成 ConfigNode
    if not isinstance(cfg, dict):  # 标量配置没有钩子可言
        return cfg
    finalize_cfg = cfg.get("finalize_cfg")  # 取配置里定义的钩子函数
    if callable(finalize_cfg):  # 存在且可调用才执行
        new_cfg = finalize_cfg(cfg)  # 执行钩子，传入完整配置
        if new_cfg is not None:  # 钩子返回了新配置则采用
            cfg = new_cfg
    return to_config_node(cfg)  # 钩子可能返回普通 dict，重新包一层 ConfigNode


# 命令行覆盖入口：把 "--opts a.b.c=1" 形式的选项按点路径逐级写入 cfg，值经 YAML 解析；
# 中间层与叶子键都必须已存在（防止拼写错误的键静默生效），全部覆盖后应用 finalize_cfg。
def parse_opts_to_config(opts, cfg):  # opts：形如 ["a.b=1"] 的列表；cfg：已加载的配置
    if not opts:  # 无覆盖项
        return finalize_config(cfg)  # 直接返回钩子处理后的配置
    for opt in opts:  # 逐条处理 "path.to.key=value"
        name, value = opt.split("=", 1)  # 按第一个等号切成键路径与原始值（值里可能含等号）
        parts = name.split(".")  # 点路径逐级下钻
        current = cfg  # 游标：当前所在层
        for part in parts[:-1]:  # 逐个中间层级
            if part not in current:  # 中间键不存在
                raise KeyError(f"Unknown config key in --opts: {name}")
            if not isinstance(current[part], dict):  # 中间层不是映射，无法继续下钻
                raise TypeError(  # 明确报错而非静默覆盖
                    f"Cannot set nested config key {name}: {part} is not a mapping."
                )
            current = current[part]  # 下钻一层
        if parts[-1] not in current:  # 叶子键必须是已存在的配置项
            raise KeyError(f"Unknown config key in --opts: {name}")
        current[parts[-1]] = to_config_node(_parse_scalar(value))  # YAML 解析值（可能是嵌套 dict）后写入
    return finalize_config(cfg)  # 覆盖完成后走钩子
