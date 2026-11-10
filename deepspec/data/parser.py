import json  # 解析样本中字符串形式的 tool_calls 字段
import re  # 正则：在整段对话文本中定位 assistant 回复片段以构造 loss_mask
import warnings  # 样本自带 system prompt 覆盖模板时发警告
from dataclasses import dataclass  # 声明式定义 ChatTemplate 结构

import torch  # 构造 loss_mask 张量

# 本模块职责：把 JSONL 对话样本渲染成模型输入文本，并切词成
# (input_ids, attention_mask, loss_mask)。loss_mask 只在 assistant 回复 token 上为 1，
# 即只对回复部分计算语言建模损失——draft model 训练只需学会模仿 assistant 输出，
# 对 user/system 部分不算损失。

@dataclass(frozen=True)
class ChatTemplate:
    # 一个模型家族对话格式的最小描述：各角色消息的前缀串、轮次结束符，
    # 以及可选的 assistant 损失前缀（gemma4 的 thought 通道标记）。
    # frozen=True：模板是全局注册表中的共享常量，禁止意外修改。
    assistant_header: str | None  # assistant 消息在文本中的起始前缀（如 "<|im_start|>assistant\n"）
    user_header: str | None  # user 消息的起始前缀
    system_prompt: str | None  # 模板内置的默认 system prompt（None 表示该模型不用）
    end_of_turn_token: str | None  # 一轮回复的结束标记（如 "<|im_end|>\n"），也是正则的终止锚点
    assistant_loss_prefix: str | None = None  # assistant 内容前的固定前缀；其 token 不计入损失


class TemplateRegistry:
    # 聊天模板注册表：名称 → ChatTemplate 的简单容器，支持运行时注册新模型家族。
    def __init__(self):
        self._templates = {}  # 内部存储：模板名 → ChatTemplate 实例

    def register(self, name, template):
        # 注册模板；重名直接断言失败，防止两个模型家族静默共用同名配置。
        assert name not in self._templates, f"Chat template {name} already exists."
        self._templates[name] = template

    def get(self, name):
        # 按名取模板；不存在时抛 KeyError，由调用方（preprocess_record）转为明确报错。
        return self._templates[name]


TEMPLATE_REGISTRY = TemplateRegistry()  # 模块级全局注册表实例，供数据管线各处共享

TEMPLATE_REGISTRY.register(  # Qwen 系（qwen2.5/qwen3 同风格 ChatML 格式）
    "qwen",
    ChatTemplate(
        assistant_header="<|im_start|>assistant\n",  # ChatML assistant 段起始
        user_header="<|im_start|>user\n",  # ChatML user 段起始
        system_prompt="You are a helpful assistant.",  # Qwen 默认系统提示
        end_of_turn_token="<|im_end|>\n",  # ChatML 段结束符
    ),
)

TEMPLATE_REGISTRY.register(  # Gemma 4 格式：无独立 system 轮，回复带 thought 通道前缀
    "gemma4",
    ChatTemplate(
        assistant_header="<|turn>model\n",  # 模型轮次前缀
        user_header="<|turn>user\n",  # 用户轮次前缀
        system_prompt=None,  # Gemma 不支持独立 system 消息
        end_of_turn_token="<turn|>\n",  # 轮次结束符
        assistant_loss_prefix="<|channel>thought\n<channel|>",  # thought 通道头：该前缀 token 不计损失
    ),
)


class GeneralParser:
    # 通用对话解析器：渲染 → 切词 → 用正则圈出 assistant 区间 → 生成 loss_mask。
    # 对话转文本与切词都委托给 tokenizer 的 chat template / 分词器本身，
    # 保证与推理时所见格式一致（draft 训练数据的分布即部署时的输入分布）。
    def __init__(self, tokenizer, chat_template):
        self.tokenizer = tokenizer  # HF tokenizer：既做 apply_chat_template 也做 encode
        self.chat_template = chat_template  # 目标家族的 ChatTemplate
        self.system_prompt = chat_template.system_prompt  # 默认系统提示（可能为 None）
        self.assistant_loss_prefix = chat_template.assistant_loss_prefix or ""  # 空串表示无损失前缀
        self.assistant_message_separator = chat_template.assistant_header or ""  # assistant 段定位锚
        # assistant 片段正则：从 "assistant_header" 之后捕获到 end_of_turn 或文本末尾为止。
        # [\s\S]*? 非贪婪跨行匹配；(?:eos|$) 允许最后一条回复没有结束符（被截断的样本）。
        # group(1) 即 assistant 内容本身（不含 header）。
        self.assistant_pattern = (
            re.escape(self.assistant_message_separator)  # header 原样匹配（转义特殊字符）
            + r"([\s\S]*?(?:"  # 捕获组 1：非贪婪的任意内容
            + re.escape(chat_template.end_of_turn_token or "")  # 到结束符为止
            + "|$))"  # 或到文本末尾
        )

    def parse(
        self,
        conversation,  # 原始对话：[{"role": ..., "content": ...}, ...] 的列表
        max_length,  # 最大 token 长度：超出即截断（loss_mask 同步按 token 下标裁剪）
    ):
        # 主入口：返回 {"input_ids", "attention_mask", "loss_mask"}。
        messages = []  # 组装后的消息列表（system + 规范化后的各轮）
        if conversation[0]["role"] == "system":
            warnings.warn(  # 样本自带 system prompt 会覆盖注册模板的默认值：提示可能的数据不一致
                "System prompt from the sample overrides the registered template.",
                stacklevel=2,
            )
            messages.append({"role": "system", "content": conversation[0]["content"]})  # 用样本自带 system
            conversation = conversation[1:]  # 其余轮次照常处理
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})  # 注入模板默认 system

        for idx, sentence in enumerate(conversation):
            role = sentence["role"]
            assert idx != 0 or role == "user", (  # 约定对话必须以 user 开头（system 已单独处理）
                f"Conversation must start with user, got {role}."
            )
            tool_calls = sentence.get("tool_calls")
            if isinstance(tool_calls, str):  # 兼容 JSONL 中 tool_calls 存成 JSON 字符串的情况
                try:
                    sentence["tool_calls"] = json.loads(tool_calls)  # 原地解析为结构化对象
                except json.JSONDecodeError:
                    assert False, f"Failed to parse tool_calls JSON: {tool_calls}"  # 坏样本直接失败，尽早暴露
            messages.append(sentence)
        render_messages = self._prepare_render_messages(messages)  # gemma4 需补 thought 前缀后再渲染
        conversation_text = render_chat_messages(  # 套用 tokenizer 自带 chat template 得到完整文本
            self.tokenizer,
            render_messages,
            add_generation_prompt=False,  # 训练样本是完整对话，不需要追加"等待生成"的提示尾
        )

        encoding = self.tokenizer(  # 一次性整段切词（与后续逐前缀切词同参数，保证前缀一致性）
            conversation_text,
            max_length=max_length,
            truncation=True,  # 超长从右侧截断
            return_tensors="pt",
            add_special_tokens=False,  # chat template 已含全部特殊 token，避免 BOS/EOS 重复
        )
        input_ids = encoding.input_ids[0]  # token 序列，形状 [L]
        attention_mask = encoding.attention_mask[0]  # 截断下全 1，占位以兼容后续 collator 接口
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)  # 初始全 0：默认不算损失

        matches = list(re.finditer(self.assistant_pattern, conversation_text, re.DOTALL))  # 所有 assistant 区间
        for match in matches:
            content_start_char = match.start(1)  # assistant 内容起始字符下标（不含 header）
            if self.assistant_loss_prefix and conversation_text.startswith(  # gemma4：跳过 thought 通道头
                self.assistant_loss_prefix,
                content_start_char,
            ):
                content_start_char += len(self.assistant_loss_prefix)  # 前缀本身不计损失
            content_end_char = match.end(1)  # 内容结束字符下标（含 end_of_turn token）
            prefix_ids = self.tokenizer.encode(  # 字符边界 → token 边界：切"内容开始前"的文本
                conversation_text[:content_start_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            full_ids = self.tokenizer.encode(  # 切"内容结束后"的文本
                conversation_text[:content_end_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            # 利用 BPE 的前缀稳定性：token 数量随文本前缀单调不减，
            # 因此两个前缀的 token 数即内容区间的 token 边界 [start, end)。
            start_token_idx = min(len(prefix_ids), len(input_ids))  # 截断后钳制到序列长度内
            end_token_idx = min(len(full_ids), len(input_ids))
            if start_token_idx < end_token_idx:
                loss_mask[start_token_idx:end_token_idx] = 1  # 该 assistant 区间计入损失

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    def _prepare_render_messages(self, messages):
        # 渲染前的消息预处理：仅 gemma4（有 assistant_loss_prefix）需要。
        # 给每个 assistant 消息补上 thought 通道前缀，使渲染文本与模型真实格式对齐；
        # 前缀随后在 parse() 里被排除出损失区间——draft 只学通道头之后的内容生成。
        if not self.assistant_loss_prefix:
            return messages  # qwen 等模板无此机制，原样返回

        render_messages = []
        for message in messages:
            if message["role"] != "assistant":
                render_messages.append(message)  # 非 assistant 消息不动
                continue

            content = message["content"]
            assert isinstance(content, str), (  # gemma4 非思考训练要求纯文本 content
                "Gemma4 non-thinking training expects assistant content to be text."
            )
            render_message = dict(message)  # 拷贝副本，避免修改调用方的原始样本
            if not content.startswith(self.assistant_loss_prefix):  # 已带前缀的不重复添加
                render_message["content"] = f"{self.assistant_loss_prefix}{content}"
            render_messages.append(render_message)
        return render_messages


def render_chat_messages(
    tokenizer,  # 提供 apply_chat_template 的 HF tokenizer
    messages,  # 规范化后的消息列表
    *,
    add_generation_prompt: bool,  # 是否在末尾追加 assistant 起始提示（推理用 True，训练用 False）
    enable_thinking: bool | None = None,  # 透传给模板的思考开关（Qwen3 等支持；None=不传）
) -> str:
    # 把消息列表渲染成最终文本：tokenize=False 先拿字符串，由调用方统一切词。
    chat_kwargs = {
        "tokenize": False,  # 只要文本不要 token，便于后续做字符级区间标注
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking  # 仅当显式指定才透传，兼容不支持该参数的模板
    return tokenizer.apply_chat_template(messages, **chat_kwargs)


def encode_chat_messages(
    tokenizer,  # HF tokenizer
    messages,  # 消息列表
    *,
    add_generation_prompt: bool,  # 同 render_chat_messages
    enable_thinking: bool | None = None,  # 同 render_chat_messages
) -> torch.LongTensor:
    # 渲染 + 切词一步到位：返回 [1, L] 的 token 张量；评估/在线采样路径使用。
    conversation_text = render_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    return tokenizer(  # add_special_tokens=False：模板已含特殊 token
        conversation_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


def preprocess_record(
    record,  # 单条 JSONL 记录，要求含 "conversations" 字段
    tokenizer,  # 用于渲染与切词
    chat_template,  # 注册表中的模板名（如 "qwen"/"gemma4"）
    max_length,  # 最大序列长度
):
    # 单样本预处理总入口：查模板 → 建解析器 → 解析为 (input_ids, attention_mask, loss_mask)。
    try:
        template = TEMPLATE_REGISTRY.get(chat_template)
    except KeyError:
        assert False, f"Unknown chat template: {chat_template}"  # 模板名写错时给出明确报错
    parser = GeneralParser(tokenizer=tokenizer, chat_template=template)
    assert "conversations" in record, "Expected `conversations` field for JSONL records."  # 字段约定校验
    return parser.parse(
        record["conversations"],
        max_length=max_length,
    )
