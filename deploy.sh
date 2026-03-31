#!/bin/bash
# DGX Spark 多节点推理 - 一键管理工具
# 用法: ./deploy.sh <命令> [选项]
# 帮助: ./deploy.sh -h

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

# ---- 帮助信息 ----
show_help() {
    cat <<EOF
${BOLD}DGX Spark 多节点推理管理工具 v${VERSION}${NC}

${BOLD}用法:${NC}
  ./deploy.sh <命令> [选项]

${BOLD}命令:${NC}
  ${GREEN}start${NC}            启动多节点推理
  ${GREEN}stop${NC}   [-y]      停止推理 (-y 跳过确认)
  ${GREEN}restart${NC} [-y]     重启推理
  ${GREEN}status${NC}           查看状态和健康检查
  ${GREEN}logs${NC}   [N] [-f]  查看节点 N 的日志 (-f 实时跟踪)
  ${GREEN}test${NC}             发送推理测试请求
  ${GREEN}check${NC}            预检: 验证 SSH/Docker/模型/网络
  ${GREEN}network${NC}          配置高速网络 (需 sudo 密码)
  ${GREEN}test-net${NC}         测试高速网络连通性
  ${GREEN}pull${NC}             在所有节点上拉取 Docker 镜像
  ${GREEN}sync${NC}             同步配置文件到所有节点

${BOLD}选项:${NC}
  -h, --help       显示此帮助
  -v, --version    显示版本号
  --dry-run        仅显示将要执行的操作，不实际执行

${BOLD}示例:${NC}
  ./deploy.sh check            # 先预检
  ./deploy.sh start            # 启动
  ./deploy.sh logs 2 -f        # 实时看节点2日志
  ./deploy.sh stop -y          # 不确认直接停止

${BOLD}首次使用:${NC}
  cp .env.example .env && vim .env   # 配置节点和模型
  ./deploy.sh check                   # 预检
  ./deploy.sh network                 # 配置光缆网络
  ./deploy.sh start                   # 启动推理
EOF
}

# ---- 参数解析 ----
DRY_RUN=false
SKIP_CONFIRM=false
FOLLOW_LOGS=false
CMD=""
CMD_ARGS=()

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)    show_help; exit 0 ;;
            -v|--version) echo "dgx-spark-dual v${VERSION}"; exit 0 ;;
            --dry-run)    DRY_RUN=true ;;
            -y|--yes)     SKIP_CONFIRM=true ;;
            -f|--follow)  FOLLOW_LOGS=true ;;
            -*)           die "未知选项: $1\n  使用 -h 查看帮助" ;;
            *)
                if [ -z "$CMD" ]; then
                    CMD="$1"
                else
                    CMD_ARGS+=("$1")
                fi
                ;;
        esac
        shift
    done
    [ -z "$CMD" ] && { show_help; exit 0; }
}

# ---- 确认提示 ----
confirm() {
    $SKIP_CONFIRM && return 0
    echo -ne "${YELLOW}$1 [y/N] ${NC}"
    read -r answer
    [[ "$answer" =~ ^[yY] ]] || { info "已取消"; exit 0; }
}

# ---- 生成 compose ----
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

    info "  接口: $node_iface → RDMA: $rdma_hca"

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

# ---- 预检 ----
cmd_check() {
    header "预检 ($NNODES 个节点)"
    local pass=0 fail=0 total=0

    _check() {
        total=$((total + 1))
        if eval "$2" &>/dev/null; then
            success "$1"; pass=$((pass + 1))
        else
            error "$1${3:+ — $3}"; fail=$((fail + 1))
        fi
    }

    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        echo ""
        info "节点 $((i+1)): $HOSTNAME ($MGMT_IP)"

        if is_local_ip "$MGMT_IP"; then
            _check "  SSH 连接 (本机)" "true"
        else
            _check "  SSH 连接" \
                "ssh -o ConnectTimeout=3 -o BatchMode=yes ${USER}@${MGMT_IP} 'echo ok'" \
                "检查 SSH 免密: ssh-copy-id ${USER}@${MGMT_IP}"
        fi

        _check "  Docker 可用" \
            "ssh_node $i 'docker info'" \
            "确认 Docker 已安装并运行"

        _check "  模型路径存在" \
            "ssh_node $i 'test -d $MODEL_PATH'" \
            "在该节点上创建/拷贝: $MODEL_PATH"

        _check "  Docker 镜像存在" \
            "ssh_node $i 'docker image inspect $SGLANG_IMAGE'" \
            "运行: ./deploy.sh pull"

        _check "  高速网 IP 已配置" \
            "ssh_node $i 'ip addr | grep -q $FAST_IP'" \
            "运行: ./deploy.sh network"

        if [ -n "$CUSTOM_SCRIPTS_DIR" ] && [ -n "$CUSTOM_ENTRYPOINT" ]; then
            _check "  自定义脚本存在" \
                "ssh_node $i 'test -f ${CUSTOM_SCRIPTS_DIR}/${CUSTOM_ENTRYPOINT}'" \
                "确认文件存在: ${CUSTOM_SCRIPTS_DIR}/${CUSTOM_ENTRYPOINT}"
        fi
    done

    # 节点间连通
    echo ""
    info "节点间连通性"
    parse_node 0; local head_ip="$FAST_IP"
    for i in $(seq 1 $((NNODES - 1))); do
        parse_node "$i"
        _check "  节点1 <-> 节点$((i+1)) (${head_ip} <-> ${FAST_IP})" \
            "ssh_node 0 'ping -c 1 -W 2 $FAST_IP'" \
            "运行: ./deploy.sh network"
    done

    echo ""
    if [ "$fail" -eq 0 ]; then
        success "全部通过 ($pass/$total) — 可以启动: ./deploy.sh start"
    else
        error "失败 $fail/$total — 请修复后重试"
    fi
    return $fail
}

# ---- 同步 ----
cmd_sync() {
    header "同步配置到 $NNODES 个节点"
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        local remote_dir
        remote_dir=$(ssh_node "$i" "echo $DEPLOY_DIR")
        info "节点 $((i+1)): $HOSTNAME ($MGMT_IP)"
        ssh_node "$i" "mkdir -p $DEPLOY_DIR/scripts"
        scp -q "$SCRIPT_DIR"/.env "$SCRIPT_DIR"/deploy.sh \
            "${USER}@${MGMT_IP}:${remote_dir}/"
        scp -q "$SCRIPT_DIR"/scripts/*.sh \
            "${USER}@${MGMT_IP}:${remote_dir}/scripts/"
    done
    success "同步完成"
}

# ---- 拉取镜像 ----
cmd_pull() {
    header "拉取镜像: $SGLANG_IMAGE"
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        info "节点 $((i+1)): $HOSTNAME"
        ssh_node "$i" "docker pull $SGLANG_IMAGE" &
    done
    wait
    success "拉取完成"
}

# ---- 启动 ----
cmd_start() {
    header "启动 ${NNODES} 节点推理"
    info "模型: $MODEL_PATH"
    info "镜像: $SGLANG_IMAGE"
    info "TP=$TP_SIZE  节点=$NNODES"

    # 清理旧容器
    for i in $(seq 0 $((NNODES - 1))); do
        local container="sglang-multinode-head"
        [ "$i" -gt 0 ] && container="sglang-multinode-worker-${i}"
        ssh_node "$i" "docker rm -f $container 2>/dev/null" || true
    done

    # 生成 compose 文件并部署
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        local compose_file="/tmp/dgx-compose-node${i}.yml"

        echo ""
        info "节点 $((i+1))/$NNODES: $HOSTNAME ($MGMT_IP) — rank $i"
        generate_compose "$i" > "$compose_file"

        if $DRY_RUN; then
            echo -e "${CYAN}--- 生成的 compose ---${NC}"
            cat "$compose_file"
            echo ""
            continue
        fi

        scp -q "$compose_file" "${USER}@${MGMT_IP}:$DEPLOY_DIR/docker-compose.node${i}.yml"
        ssh_node "$i" "cd $DEPLOY_DIR && docker compose -f docker-compose.node${i}.yml up -d"

        [ "$i" -eq 0 ] && sleep 5
        rm -f "$compose_file"
    done

    $DRY_RUN && { info "[dry-run] 未实际执行"; return; }

    echo ""
    success "容器已启动，等待模型加载..."

    # 轮询健康检查 (最多等 5 分钟)
    info "轮询 /health (最多 5 分钟，Ctrl+C 跳过)..."
    local deadline=$((SECONDS + 300))
    while [ $SECONDS -lt $deadline ]; do
        if ssh_node 0 "curl -sf http://localhost:${SGLANG_PORT}/health" &>/dev/null; then
            echo ""
            success "服务就绪! API: http://${MGMT_IP}:${SGLANG_PORT}"
            return
        fi
        echo -n "."
        sleep 5
    done
    echo ""
    warn "等待超时，服务可能仍在加载，使用 ./deploy.sh logs 查看进度"
}

# ---- 停止 ----
cmd_stop() {
    header "停止多节点推理"
    confirm "确认停止所有 $NNODES 个节点?"

    for i in $(seq $((NNODES - 1)) -1 0); do
        parse_node "$i"
        info "停止 $HOSTNAME..."
        ssh_node "$i" "cd $DEPLOY_DIR && docker compose -f docker-compose.node${i}.yml down 2>/dev/null" || true
    done
    success "已停止"
}

# ---- 重启 ----
cmd_restart() {
    confirm "确认重启所有 $NNODES 个节点?"
    SKIP_CONFIRM=true  # stop 内部不再重复确认
    cmd_stop
    sleep 2
    cmd_start
}

# ---- 状态 ----
cmd_status() {
    header "集群状态"
    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        echo ""
        info "节点 $((i+1)): $HOSTNAME ($MGMT_IP)"
        ssh_node "$i" "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' --filter name=sglang-multinode" 2>/dev/null \
            || warn "无法连接"
    done

    echo ""
    parse_node 0
    if ssh_node 0 "curl -sf --max-time 5 http://localhost:${SGLANG_PORT}/health" &>/dev/null; then
        success "API 正常: http://${MGMT_IP}:${SGLANG_PORT}"
    else
        warn "API 未就绪 (模型可能还在加载)"
    fi
}

# ---- 日志 ----
cmd_logs() {
    local idx="${CMD_ARGS[0]:-1}"

    # 转为 0-based
    if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ]; then
        idx=$((idx - 1))
    else
        idx=0
    fi
    [ "$idx" -ge "$NNODES" ] && die "只有 $NNODES 个节点"

    parse_node "$idx"
    local container="sglang-multinode-head"
    [ "$idx" -gt 0 ] && container="sglang-multinode-worker-${idx}"

    header "节点 $((idx+1)) ($HOSTNAME) 日志"
    if $FOLLOW_LOGS; then
        ssh_node "$idx" "docker logs -f --tail 50 $container 2>&1"
    else
        ssh_node "$idx" "docker logs --tail 80 $container 2>&1"
    fi
}

# ---- 测试推理 ----
cmd_test() {
    parse_node 0
    header "测试推理 -> $HOSTNAME:$SGLANG_PORT"
    local result
    result=$(ssh_node 0 "curl -s --max-time 120 http://localhost:${SGLANG_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"/model\", \"messages\": [{\"role\": \"user\", \"content\": \"你好，请用一句话介绍自己\"}], \"max_tokens\": 200}'" 2>&1)

    if echo "$result" | python3 -m json.tool 2>/dev/null; then
        success "推理正常"
    else
        error "请求失败: $result"
        echo ""
        info "可能原因: 服务未就绪，运行 ./deploy.sh status 检查"
    fi
}

# ---- 配置网络 ----
cmd_network() {
    header "配置高速网络"

    if $DRY_RUN; then
        for i in $(seq 0 $((NNODES - 1))); do
            parse_node "$i"
            info "[dry-run] 将在 $HOSTNAME ($MGMT_IP) 上配置 IP=$FAST_IP MTU=$FAST_MTU"
        done
        return
    fi

    cmd_sync

    echo "请输入机器的 sudo 密码:"
    read -rs SUDO_PASS
    echo ""
    [ -z "$SUDO_PASS" ] && die "密码不能为空"

    for i in $(seq 0 $((NNODES - 1))); do
        parse_node "$i"
        info "节点 $((i+1)): $HOSTNAME ($MGMT_IP) → $FAST_IP"
        ssh_node "$i" "echo '$SUDO_PASS' | sudo -S bash $DEPLOY_DIR/scripts/setup-network.sh $((i+1))" 2>&1 | grep -v "密码"
        echo ""
    done
    success "网络配置完成"
}

# ---- 测试网络 ----
cmd_test_net() {
    header "测试高速网络"
    cmd_sync

    parse_node 0
    info "在 $HOSTNAME 上测试连接..."
    ssh_node 0 "bash $DEPLOY_DIR/scripts/test-connection.sh 1 2"
}

# ---- 主入口 ----
parse_args "$@"

# 除 help/version 外的命令都需要加载 .env
load_env

# 获取 head 节点 IP
parse_node 0
HEAD_FAST_IP="$FAST_IP"

case "$CMD" in
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    restart)  cmd_restart ;;
    status)   cmd_status ;;
    logs)     cmd_logs ;;
    test)     cmd_test ;;
    check)    cmd_check ;;
    network)  cmd_network ;;
    test-net) cmd_test_net ;;
    pull)     cmd_pull ;;
    sync)     cmd_sync ;;
    *)        error "未知命令: $CMD"; echo ""; show_help; exit 1 ;;
esac
