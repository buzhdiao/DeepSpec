#!/usr/bin/env bash
# 按 GPU 数拉起多台独立 sglang OpenAI 兼容 server（每卡一台，各自占一个 HTTP 端口），
# 供 generate_train_data.py 并发重生成训练数据使用；附带心跳打印与退出清理逻辑。
set -euo pipefail  # 严格模式：未定义变量报错、任一命令失败即退出、管道中段失败也视为失败

# Requires SGLang to be installed separately (not in requirements.txt):
#   pip install "sglang[all]"
# See https://docs.sglang.ai/get_started/install.html for details.

model_path=Qwen/Qwen3-4B  # 待部署模型：HF 名称或本地权重路径
num_workers=8  # server 实例数 = GPU 数（每卡一台）
start_port=30000  # 首台 server 的 HTTP 端口，后续依次 +1
start_nccl_port=31000  # 首台 server 的 NCCL 端口，后续依次 +1（避免各实例端口冲突）
host=0.0.0.0  # 监听地址：0.0.0.0 允许本机以外访问
dtype=bfloat16  # 权重加载精度
mem_frac=0.9  # 显存预算比例：预分配给 KV cache 的静态显存占比
log_dir=logs/sglang_qwen3_4b  # 各 worker 日志目录
heartbeat_interval=300  # 心跳打印间隔（秒）

get_host_ip() {
    # 探测本机对外 IP：依次尝试 hostname -I、ip route，全失败则回退 127.0.0.1。
    local host_ip=""

    if command -v hostname > /dev/null 2>&1; then  # hostname 命令存在时
        host_ip=$(hostname -I 2> /dev/null | awk '{print $1}')  # 取 -I 列出的第一个 IP
    fi

    if [[ -z "${host_ip}" ]] && command -v ip > /dev/null 2>&1; then  # hostname 不可用/无输出：用 ip 路由兜底
        # 查到公网地址 1.1.1.1 的路由，取其 src 源 IP（即本机对外 IP）
        host_ip=$(
            ip -4 route get 1.1.1.1 2> /dev/null | awk '
                /src/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "src") {
                            print $(i + 1)
                            exit
                        }
                    }
                }
            '
        )
    fi

    if [[ -z "${host_ip}" ]]; then
        host_ip=127.0.0.1  # 都失败：回退本机回环地址
    fi

    printf '%s\n' "${host_ip}"
}

mkdir -p "${log_dir}"  # 日志目录不存在则创建（幂等）

host_ip=$(get_host_ip)
pids=()  # 各 worker 的后台进程 pid
ports=()  # 各 worker 的 HTTP 端口
nccl_ports=()  # 各 worker 的 NCCL 端口
heartbeat_pid=""  # 心跳循环的后台 pid

print_heartbeat() {
    # 打印一次心跳：逐个 worker 用 kill -0 检查进程存活并统计存活数。
    local timestamp alive_count status pid port nccl_port
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    alive_count=0

    echo "[${timestamp}] heartbeat"
    for ((idx = 0; idx < ${#pids[@]}; idx++)); do  # 遍历每个 worker
        pid=${pids[$idx]}
        port=${ports[$idx]}
        nccl_port=${nccl_ports[$idx]}
        status=dead
        if kill -0 "${pid}" > /dev/null 2>&1; then  # kill -0 只探测进程存在，不发送信号
            status=alive
            alive_count=$((alive_count + 1))
        fi
        echo "  worker_index=${idx} pid=${pid} port=${port} nccl_port=${nccl_port} status=${status}"
    done
    echo "  alive_workers=${alive_count}/${#pids[@]}"
}

heartbeat_loop() {
    # 心跳主循环：每个周期打印一次存活状态，直到被清理逻辑终止。
    while true; do
        sleep "${heartbeat_interval}"
        print_heartbeat
    done
}

cleanup() {
    # 退出/中断清理：先杀心跳循环，再杀全部 sglang worker，最后 wait 回收，防止残留进程占卡。
    if [[ -n "${heartbeat_pid}" ]] && kill -0 "${heartbeat_pid}" > /dev/null 2>&1; then
        kill "${heartbeat_pid}" > /dev/null 2>&1 || true  # 终止心跳循环
    fi
    for pid in "${pids[@]:-}"; do  # 数组可能为空：${pids[@]:-} 兼容 set -u
        if kill -0 "${pid}" > /dev/null 2>&1; then
            kill "${pid}" > /dev/null 2>&1 || true  # 尽力杀掉每台 server；失败不阻断清理
        fi
    done
    wait || true  # 等待后台进程退出（信号场景下 wait 可能返回非零，忽略）
}

trap cleanup INT TERM EXIT  # Ctrl-C / kill / 正常退出三条路径都执行清理

for ((gpu_id = 0; gpu_id < num_workers; gpu_id++)); do  # 逐卡拉起一台 server
    port=$((start_port + gpu_id))  # 本实例 HTTP 端口
    nccl_port=$((start_nccl_port + gpu_id))  # 本实例 NCCL 端口
    log_file=${log_dir}/worker_${host_ip}_gpu_${gpu_id}_port_${port}.log

    echo "Starting sglang worker ip=${host_ip} gpu=${gpu_id} port=${port} nccl_port=${nccl_port} log=${log_file}"
    # CUDA_VISIBLE_DEVICES 让本实例只绑定一张卡；"$@" 把脚本调用者的额外参数原样透传给 sglang serve；
    # 各 --flag 含义见上方变量定义；stdout/stderr 重定向到日志文件，后台运行
    CUDA_VISIBLE_DEVICES=${gpu_id} sglang serve \
        --model-path "${model_path}" \
        --host "${host}" \
        --port "${port}" \
        --nccl-port "${nccl_port}" \
        --dtype "${dtype}" \
        --mem-fraction-static "${mem_frac}" \
        "$@" > "${log_file}" 2>&1 &
    pids+=("$!")  # 记录后台进程 pid
    ports+=("${port}")
    nccl_ports+=("${nccl_port}")
done

echo "Workers launched:"
for ((gpu_id = 0; gpu_id < num_workers; gpu_id++)); do  # 打印全部 worker 的访问地址便于核对
    port=$((start_port + gpu_id))
    nccl_port=$((start_nccl_port + gpu_id))
    echo "  http://${host_ip}:${port} nccl_port=${nccl_port}"
done
echo "Heartbeat interval: ${heartbeat_interval}s"
print_heartbeat
heartbeat_loop &  # 心跳循环放后台，不阻塞主流程
heartbeat_pid=$!

wait "${pids[@]}"  # 阻塞等待全部 server 进程退出（任一崩溃会触发 trap 清理其余进程）
