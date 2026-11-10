import argparse  # 命令行参数解析
import json  # 读写 JSONL（输入样本、输出/错误文件）
import os  # 判断输出文件是否存在（断点续跑）
import random  # 随机抽取 gpt-oss 的 reasoning effort
import time  # 服务器探活计时、调度空转 sleep
from concurrent.futures import ThreadPoolExecutor, as_completed  # 线程池并发请求与完成通知

from openai import OpenAI  # OpenAI 兼容客户端：sglang server 即以 /v1 接口暴露
from tqdm import tqdm  # 处理进度条


def parse_args():
    # 解析重生成参数：模型、服务器地址、输入/输出文件与采样选项。
    parser = argparse.ArgumentParser(
        description="Regenerate JSONL conversations through OpenAI-compatible sglang servers."
    )
    parser.add_argument("--model", required=True)  # 生成用模型名（须与 server 加载的模型一致）
    parser.add_argument("--server-address", nargs="+", required=True)  # 一个或多个 server 地址（host:port），请求轮询分流
    parser.add_argument("--input-file-path", required=True)  # 输入 JSONL：每行含 conversations 的对话样本
    parser.add_argument("--output-file-path", required=True)  # 成功结果输出 JSONL
    parser.add_argument("--concurrency", type=int, default=64)  # 单 server 在途请求上限（总线程数 = 并发 × server 数）
    parser.add_argument("--temperature", type=float, default=0.7)  # 采样温度
    parser.add_argument("--top-p", type=float, default=None)  # nucleus 采样；None 表示不传
    parser.add_argument("--top-k", type=int, default=None)  # top-k 采样；sglang 扩展参数，走 extra_body
    parser.add_argument("--min-p", type=float, default=None)  # min-p 采样；sglang 扩展参数，走 extra_body
    parser.add_argument("--repetition-penalty", type=float, default=None)  # 重复惩罚；sglang 侧映射为 presence_penalty
    parser.add_argument("--max-tokens", type=int, default=4096)  # 单次生成的最大 token 数
    parser.add_argument("--num-samples", type=int, default=None)  # 本次最多处理多少行；None 表示全部
    parser.add_argument("--resume", action="store_true")  # 断点续跑：按已有输出行数跳过已处理样本
    parser.add_argument("--is-reasoning-model", action="store_true")  # 推理模型：额外保存 reasoning_content 到 thinking 字段
    parser.add_argument("--is-gpt-oss", action="store_true")  # gpt-oss 模型：每次请求随机 reasoning_effort
    # 以下两个互斥：控制 chat template 的 thinking 开关（Qwen3 等支持）
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", action="store_true")  # extra_body.chat_template_kwargs.enable_thinking=True
    thinking_group.add_argument("--disable-thinking", action="store_true")  # extra_body.chat_template_kwargs.enable_thinking=False
    return parser.parse_args()


def validate_args(args):
    # 采样参数合法性校验：越界直接报错，避免把非法参数静默发给 server。
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    if args.top_p is not None and not 0.0 <= args.top_p <= 1.0:
        raise ValueError("top-p must be between 0.0 and 1.0")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("top-k must be greater than 0")
    if args.min_p is not None and not 0.0 <= args.min_p <= 1.0:
        raise ValueError("min-p must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")


def get_random_reasoning_effort():
    # gpt-oss 专用：按 4:4:2 权重随机选 reasoning effort，制造训练数据的多档思考深度。
    return random.choices(["low", "medium", "high"], weights=[4, 4, 2], k=1)[0]


def compute_context_length(conversations):
    # 估算整段对话的上下文长度（按空白分词数，粗略近似 token 数），用于统计。
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            length += len(content.split())  # 纯文本消息：按空白分词计数
        elif isinstance(content, list):
            for part in content:  # 多模态消息：只累计其中的 text 段
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    length += len(part["text"].split())
    return length


def build_query_kwargs(args, messages, max_tokens=None):
    # 组装 chat.completions.create 的请求参数：OpenAI 标准字段 + sglang 扩展 extra_body。
    query_kwargs = {
        "model": args.model,  # server 侧已加载的模型名
        "messages": messages,  # 截至当前 user 轮的消息列表（不含旧 assistant 回复）
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,  # 探活时可传小值
        "temperature": args.temperature,
        "stream": False,  # 非流式：一次性取回完整回复
    }
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty  # sglang 把重复惩罚落到 presence_penalty 字段

    extra_body = {}  # 非 OpenAI 标准的字段统一放 extra_body
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.enable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True  # 渲染 chat template 时开启思考
    if args.disable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False  # 渲染 chat template 时关闭思考
    if extra_body:
        query_kwargs["extra_body"] = extra_body

    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()  # gpt-oss：随机思考强度
    return query_kwargs


def error_sample(sample, message):
    # 给样本打上错误标记（不抛异常，错误随结果写 _error.jsonl 供续跑与排查）。
    sample["status"] = "error"
    sample["error"] = message
    return sample


def call_sglang(args, server_address, sample, max_tokens=None):
    # 对单个样本做多轮重生成：保留 system 与 user 轮原文，逐轮请求 server 重新生成 assistant 回复。
    conversations = sample.get("conversations")
    if not conversations:
        return error_sample(sample, "Missing conversations")
    if conversations[0].get("role") == "assistant":
        return error_sample(sample, "Data starts with an assistant message")  # 首条是 assistant 无法发起请求，判为脏数据

    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")  # sglang 兼容 OpenAI 接口；key 随意填
    regenerated = []

    for message in conversations:
        role = message.get("role")
        if role == "system":
            regenerated.append(message)  # system 提示原样保留在上下文中
            continue
        if role == "assistant":
            continue  # 旧 assistant 回复整体丢弃：重生成时不再喂给模型，避免污染新回复
        if role != "user":
            return error_sample(sample, f"Invalid message role: {role}")

        regenerated.append(message)  # 追加 user 轮后，请求生成对应的 assistant 回复
        try:
            response = client.chat.completions.create(
                **build_query_kwargs(args, regenerated, max_tokens=max_tokens)
            )
        except Exception as exc:
            return error_sample(sample, str(exc))  # 请求失败：整条样本判错，不做单轮重试

        response_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,  # 新生成的回复正文
        }
        if args.is_reasoning_model:
            response_message["thinking"] = response.choices[0].message.reasoning_content  # 推理模型额外保存思考内容
        regenerated.append(response_message)  # 新回复进入上下文，继续处理下一轮

    sample["conversations"] = regenerated  # 用重生成后的完整对话覆盖原样本
    sample["status"] = "success"
    return sample


def count_lines(path):
    # 统计文件行数（断点续跑与进度条总数都要用）。
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def find_resume_offset(output_path, error_path):
    # 计算续跑偏移：成功数 + 错误数 = 输入文件中已处理行数；返回 (跳过行数, 成功数, 错误数)。
    if not os.path.exists(output_path):
        return 0, 0, 0  # 无历史输出：从头开始

    success_count = count_lines(output_path)  # 已成功写入的行数
    error_count = count_lines(error_path) if os.path.exists(error_path) else 0  # 已判错的行数
    return success_count + error_count, success_count, error_count


def validate_server(args, server_address, probe):
    # 探活单台 server：发一条 max_tokens=1 的真实请求，返回 (地址, 结果, 耗秒)。
    start_time = time.perf_counter()
    try:
        result = call_sglang(args, server_address, dict(probe), max_tokens=1)  # 用探针样本走完整调用链
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}  # call_sglang 一般不抛，这里兜底
    elapsed = time.perf_counter() - start_time
    return server_address, result, elapsed


def validate_servers(args):
    # 并行探活全部 server：可用者进入轮询列表；全部不可用则直接失败。
    probe = {"conversations": [{"role": "user", "content": "Hello"}]}  # 最小合法对话作探针
    server_results = {}  # 地址 → 是否可用
    server_count = len(args.server_address)

    print(f"Validating {server_count} sglang servers in parallel...", flush=True)
    with ThreadPoolExecutor(max_workers=server_count) as executor:  # 每台 server 一个线程同时探活
        future_to_server = {
            executor.submit(
                validate_server,
                args,
                server_address,
                probe,
            ): server_address
            for server_address in args.server_address
        }

        for completed_count, future in enumerate(as_completed(future_to_server), 1):
            server_address = future_to_server[future]
            try:
                _, result, elapsed = future.result()
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}  # 线程内异常也按探活失败处理
                elapsed = 0.0

            if result.get("status") == "success":
                server_results[server_address] = True
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"ok server {server_address} elapsed={elapsed:.2f}s",
                    flush=True,
                )
            else:
                server_results[server_address] = False
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"skip server {server_address} elapsed={elapsed:.2f}s: "
                    f"{result.get('error')}",
                    flush=True,
                )

    valid_servers = [
        server_address
        for server_address in args.server_address
        if server_results[server_address]
    ]
    invalid_servers = [
        server_address
        for server_address in args.server_address
        if not server_results[server_address]
    ]

    print(f"Available servers ({len(valid_servers)}/{server_count}): {valid_servers}")
    print(
        f"Unavailable servers ({len(invalid_servers)}/{server_count}): "
        f"{invalid_servers}"
    )
    if not valid_servers:
        raise RuntimeError("No available sglang server")  # 一台都不可用：无法继续
    return valid_servers


def write_finished_result(
    future,
    output_handle,
    error_handle,
    stats,
):
    # 收割一个已完成请求：成功行写输出文件并累计上下文统计，失败行写错误文件。
    sample = future.result()
    if sample["status"] == "error":
        error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        stats["errors"] += 1
        return

    context_length = compute_context_length(sample.get("conversations", []))  # 估算重生成后的上下文长度
    stats["context_sum"] += context_length
    stats["context_min"] = (
        context_length
        if stats["context_min"] is None
        else min(stats["context_min"], context_length)  # 首个成功样本直接赋初值
    )
    stats["context_max"] = max(stats["context_max"], context_length)
    stats["success"] += 1
    output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_config(args):
    # 打印本次运行的关键配置，便于日志留存比对。
    print("Configuration:")
    print(f"  model: {args.model}")
    print(f"  servers: {args.server_address}")
    print(f"  input: {args.input_file_path}")
    print(f"  output: {args.output_file_path}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  min_p: {args.min_p}")
    print(f"  resume: {args.resume}")


def main():
    # 主流程：解析校验 → 计算续跑偏移 → 探活 → 按 server 轮询提交请求并限流 → 收尾收割与统计。
    args = parse_args()
    validate_args(args)
    print_config(args)

    total_lines = count_lines(args.input_file_path)  # 输入总行数
    error_path = args.output_file_path.replace(".jsonl", "_error.jsonl")  # 错误输出路径：与成功输出同目录
    skip_lines, existing_success, existing_errors = (  # 续跑模式下跳过已处理的行
        find_resume_offset(args.output_file_path, error_path)
        if args.resume
        else (0, 0, 0)
    )
    if skip_lines >= total_lines:
        print(f"All {total_lines} samples are already processed.")  # 全部处理过：直接结束
        return

    if args.resume and skip_lines > 0:
        print(
            "Resume mode: "
            f"{existing_success} success, {existing_errors} errors, skip {skip_lines}"
        )

    valid_servers = validate_servers(args)
    print(f"Using servers: {valid_servers}")

    file_mode = "a" if args.resume and skip_lines > 0 else "w"  # 续跑追加写，否则覆盖写
    stats = {
        "success": 0,  # 本次成功条数
        "errors": 0,  # 本次失败条数
        "context_sum": 0,  # 成功样本上下文长度总和
        "context_min": None,  # 上下文长度最小值（首个样本后再取 min）
        "context_max": 0,  # 上下文长度最大值
    }
    queues = {server_address: [] for server_address in valid_servers}  # 各 server 的在途请求队列
    next_server_index = 0  # 轮询指针：样本依次分发到各 server
    submitted_count = 0  # 已提交（读取并调度）的样本数

    with (
        open(args.input_file_path, "r", encoding="utf-8") as input_handle,
        open(args.output_file_path, file_mode, encoding="utf-8") as output_handle,
        open(error_path, file_mode, encoding="utf-8") as error_handle,
        ThreadPoolExecutor(max_workers=args.concurrency * len(valid_servers)) as executor,  # 线程池上限 = 单机并发 × server 数
    ):
        for _ in range(skip_lines):
            next(input_handle, None)  # 续跑：逐行跳过已处理的输入行

        progress_total = (  # 进度条终点：全量或 num-samples 截断后的总量
            total_lines
            if args.num_samples is None
            else min(total_lines, skip_lines + args.num_samples)
        )
        progress = tqdm(total=progress_total, initial=skip_lines, desc="Processing")
        for line in input_handle:
            if args.num_samples is not None and submitted_count >= args.num_samples:
                break  # 达到本次处理上限

            sample = json.loads(line)
            server_address = valid_servers[next_server_index]  # 轮询选一台 server
            next_server_index = (next_server_index + 1) % len(valid_servers)

            while len(queues[server_address]) >= args.concurrency:  # 该 server 在途请求满：先收割已完成的
                wrote_result = False
                for future in list(queues[server_address]):
                    if future.done():
                        write_finished_result(
                            future, output_handle, error_handle, stats
                        )
                        queues[server_address].remove(future)  # 收割后腾出并发名额
                        wrote_result = True
                        break  # 每次只收割一个，重新检查额度
                if not wrote_result:
                    time.sleep(0.05)  # 暂无可收割：小睡后重试，避免忙等烧 CPU

            future = executor.submit(call_sglang, args, server_address, sample)  # 提交单样本重生成任务
            queues[server_address].append(future)
            submitted_count += 1
            progress.update(1)

        for server_address in valid_servers:  # 输入读完：收割各 server 剩余在途请求
            for future in queues[server_address]:
                write_finished_result(future, output_handle, error_handle, stats)
        progress.close()

    print("Processing completed.")
    print(f"  success: {stats['success']}")
    print(f"  errors: {stats['errors']}")
    if stats["success"] > 0:
        avg_context = stats["context_sum"] / stats["success"]
        print(f"  context_min: {stats['context_min']}")
        print(f"  context_max: {stats['context_max']}")
        print(f"  context_avg: {avg_context:.2f}")  # 上下文长度均值：粗估训练样本长度分布


if __name__ == "__main__":
    main()
