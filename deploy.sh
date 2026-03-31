#!/bin/bash
# DGX Spark 多节点推理 - 一键管理脚本
# 用法: bash deploy.sh <start|stop|restart|status|logs|test|network|sync>

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

DEPLOY_DIR="~/dgx-spark-multinode"
NNODES=${#NODE_LIST[@]}

# ConnectX-7 网口名列表 (DGX Spark 上的 4 个口)
CX7_IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

# 解析节点信息: parse_node <index> -> MGMT_IP FAST_IP USER HOSTNAME FAST_IFACE
parse_node() {
    local _iface
    IFS=',' read -r MGMT_IP FAST_IP USER HOSTNAME _iface <<< "${NODE_LIST[$1]}"
    FAST_IFACE="${_iface}"
}

# SSH 到指定节点
ssh_node() {
    local idx=$1; shift
    parse_node "$idx"
    ssh -o ConnectTimeout=5 "${USER}@${MGMT_IP}" "$@"
}

# 远程自动检测有光缆的 ConnectX-7 口
# 用法: detect_iface <node_index>  -> 结果写入 FAST_IFACE
detect_iface() {
    local idx=$1
    parse_node "$idx"
    # 如果 NODE_LIST 中已指定接口，直接使用
    if [ -n "$FAST_IFACE" ]; then
        return
    fi
    # SSH 到目标机器，找第一个 carrier=1 的 ConnectX-7 口
    local detected
    detected=$(ssh_node "$idx" "for dev in $CX7_IFACES; do [ -f /sys/class/net/\$dev/carrier ] && [ \$(cat /sys/class/net/\$dev/carrier 2>/dev/null) = 1 ] && echo \$dev && break; done" 2>/dev/null)
    if [ -z "$detected" ]; then
        echo "警告: 节点 $HOSTNAME ($MGMT_IP) 未检测到有光缆的 ConnectX-7 口" >&2
        FAST_IFACE="enp1s0f0np0"  # fallback
    else
        FAST_IFACE="$detected"
    fi
}

# 从网口名推导 RDMA HCA 名: enp1s0f1np1 -> rocep1s0f1
iface_to_rdma() {
    echo "roce$(echo "$1" | sed 's/np[0-9]*$//' | sed 's/^en//')"
}

# 获取 head 节点的高速网 IP
parse_node 0
HEAD_FAST_IP="$FAST_IP"

# ---- 同步 ----
sync_files() {
    echo "=== 同步配置到 $NNODES 个节点 ==="
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        echo "  -> $HOSTNAME ($MGMT_IP)"
        ssh -o ConnectTimeout=5 "${USER}@${MGMT_IP}" "mkdir -p $DEPLOY_DIR"
        scp -q "$SCRIPT_DIR"/.env "$SCRIPT_DIR"/*.yml "$SCRIPT_DIR"/*.sh \
            "${USER}@${MGMT_IP}:$DEPLOY_DIR/"
    done
    echo "同步完成"
}

# ---- 生成 compose 文件 ----
generate_compose() {
    local rank=$1
    detect_iface "$rank"
    local role="head"
    local container="sglang-multinode-head"
    [ "$rank" -gt 0 ] && role="worker-${rank}" && container="sglang-multinode-worker-${rank}"

    local quant_arg=""
    [ -n "$QUANTIZATION" ] && quant_arg="--quantization $QUANTIZATION"

    local node_iface="$FAST_IFACE"
    local rdma_hca
    rdma_hca="$(iface_to_rdma "$node_iface")"

    echo "    接口: $node_iface (RDMA: $rdma_hca)" >&2

    # 自定义 patch 挂载
    local volumes_extra=""
    local entrypoint_line=""
    if [ -n "$CUSTOM_SCRIPTS_DIR" ] && [ -n "$CUSTOM_ENTRYPOINT" ]; then
        volumes_extra="      - ${CUSTOM_SCRIPTS_DIR}/${CUSTOM_REASONING_DETECTOR}:/custom/${CUSTOM_REASONING_DETECTOR}
      - ${CUSTOM_SCRIPTS_DIR}/${CUSTOM_ENTRYPOINT}:/custom/entrypoint.sh"
        entrypoint_line='    entrypoint: ["/custom/entrypoint.sh"]'
    fi

    cat <<YAML
services:
  sglang-${role}:
    image: ${SGLANG_IMAGE}
    container_name: ${container}
    restart: unless-stopped
    network_mode: host
    ipc: host
    shm_size: "32g"
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - ${MODEL_PATH}:/model:ro
${volumes_extra}
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NCCL_SOCKET_IFNAME=${node_iface}
      - NCCL_IB_DISABLE=0
      - NCCL_NET_GDR_LEVEL=5
      - NCCL_IB_HCA=${rdma_hca}
      - NCCL_DEBUG=WARN
      - GLOO_SOCKET_IFNAME=${node_iface}
      - MASTER_ADDR=${HEAD_FAST_IP}
      - MASTER_PORT=${NCCL_INIT_PORT}
${entrypoint_line}
    command: >
      python3 -m sglang.launch_server
      --model-path /model
      --tp ${TP_SIZE}
      --nnodes ${NNODES}
      --node-rank ${rank}
      --nccl-init-addr ${HEAD_FAST_IP}:${NCCL_INIT_PORT}
      --host 0.0.0.0
      --port ${SGLANG_PORT}
      --attention-backend ${ATTENTION_BACKEND}
      --mem-fraction-static ${MEM_FRACTION}
      ${quant_arg}
      ${EXTRA_ARGS}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
YAML
}

# ---- 启动 ----
start() {
    echo "=== 启动 ${NNODES} 节点推理 ==="
    echo "模型: $MODEL_PATH"
    echo "镜像: $SGLANG_IMAGE"
    echo "TP: $TP_SIZE, 节点数: $NNODES"
    echo ""

    # 停旧容器
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        local container="sglang-multinode-head"
        [ "$i" -gt 0 ] && container="sglang-multinode-worker-${i}"
        ssh_node "$i" "docker rm -f $container 2>/dev/null" || true
    done

    # 生成 compose 文件并部署
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        local compose_file="/tmp/docker-compose.node${i}.yml"
        echo ">>> 节点 $((i+1))/$NNODES: $HOSTNAME ($MGMT_IP) — rank $i"
        generate_compose "$i" > "$compose_file"

        scp -q "$compose_file" "${USER}@${MGMT_IP}:$DEPLOY_DIR/docker-compose.node${i}.yml"
        ssh_node "$i" "cd $DEPLOY_DIR && docker compose -f docker-compose.node${i}.yml up -d"

        # head 启动后等几秒再启动 worker
        [ "$i" -eq 0 ] && sleep 5
    done

    echo ""
    echo "启动完成，等待模型加载 (约2-3分钟)..."
    echo "  查看日志: bash deploy.sh logs [节点号]"
    echo "  测试请求: bash deploy.sh test"
}

# ---- 停止 ----
stop() {
    echo "=== 停止多节点推理 ==="
    # 先停 worker 再停 head
    for i in $(seq $((NNODES - 1)) -1 0); do
        parse_node "$i"
        echo "  停止 $HOSTNAME..."
        ssh_node "$i" "cd $DEPLOY_DIR && docker compose -f docker-compose.node${i}.yml down 2>/dev/null" || true
    done
    echo "已停止"
}

# ---- 重启 ----
restart() {
    stop
    sleep 2
    start
}

# ---- 状态 ----
status() {
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        echo "=== 节点 $((i+1)): $HOSTNAME ($MGMT_IP) ==="
        ssh_node "$i" "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' --filter name=sglang-multinode" 2>/dev/null || echo "  无法连接"
        echo ""
    done

    echo "=== API 健康检查 ==="
    parse_node 0
    ssh_node 0 "curl -s --max-time 5 http://localhost:${SGLANG_PORT}/health 2>/dev/null" \
        && echo " — 正常" \
        || echo "未就绪 (模型可能还在加载)"
}

# ---- 日志 ----
logs() {
    local idx=${2:-0}
    # 用户输入的是 1-based，内部 0-based
    if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] 2>/dev/null; then
        idx=$((idx - 1))
    else
        idx=0
    fi

    if [ "$idx" -ge "$NNODES" ]; then
        echo "错误: 只有 $NNODES 个节点"; exit 1
    fi

    parse_node "$idx"
    local container="sglang-multinode-head"
    [ "$idx" -gt 0 ] && container="sglang-multinode-worker-${idx}"

    echo "=== 节点 $((idx+1)) ($HOSTNAME) 日志 ==="
    ssh_node "$idx" "docker logs --tail 80 $container 2>&1"
}

# ---- 测试 ----
test_inference() {
    parse_node 0
    echo "=== 测试请求 -> $HOSTNAME:$SGLANG_PORT ==="
    ssh_node 0 "curl -s --max-time 120 http://localhost:${SGLANG_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"/model\", \"messages\": [{\"role\": \"user\", \"content\": \"你好，请用一句话介绍自己\"}], \"max_tokens\": 200}'" \
        | python3 -m json.tool 2>/dev/null \
        || echo "请求失败，服务可能未就绪"
}

# ---- 网络配置 ----
setup_network() {
    echo "请输入机器的 sudo 密码:"
    read -rs SUDO_PASS
    echo ""

    sync_files

    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        echo ">>> 配置节点 $((i+1)): $HOSTNAME ($MGMT_IP) -> $FAST_IP"
        ssh_node "$i" "echo '$SUDO_PASS' | sudo -S bash $DEPLOY_DIR/setup-network.sh $((i+1))" 2>&1 | grep -v "密码"
        echo ""
    done
}

# ---- 入口 ----
case "${1:-help}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    logs "$@" ;;
    test)    test_inference ;;
    network) setup_network ;;
    sync)    sync_files ;;
    *)
        echo "DGX Spark 多节点推理管理工具"
        echo ""
        echo "用法: bash deploy.sh <命令>"
        echo ""
        echo "命令:"
        echo "  start       启动多节点推理"
        echo "  stop        停止多节点推理"
        echo "  restart     重启"
        echo "  status      查看状态"
        echo "  logs [N]    查看节点 N 的日志 (默认 1)"
        echo "  test        发送测试请求"
        echo "  network     配置高速网络 (需 sudo)"
        echo "  sync        同步配置到所有节点"
        ;;
esac
