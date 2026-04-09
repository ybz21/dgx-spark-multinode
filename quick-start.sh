#!/bin/bash
# ============================================================
# DGX Spark 双节点一键部署 (vLLM + Ray)
#
# 只需配置: 工作节点IP、Docker镜像、模型路径
# 自动完成: 校验 → 传输镜像 → 传输模型 → 启动双节点
#
# 用法:
#   bash quick-start.sh WORKER_IP MODEL_PATH [--image IMAGE]
#   bash quick-start.sh 192.168.130.8 ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603 ~/models/Qwen3___5-122B-A10B-NVFP4
#
# 选项:
#   --no-sync-model   跳过模型同步
#   --no-sync-image   跳过镜像同步
#   --dry-run         仅校验，不实际执行
#   --stop            停止当前服务
#   --status          查看服务状态
#   --port PORT       API 端口 (默认 30000)
#   --max-len LEN     最大上下文长度 (默认 8192)
# ============================================================

set -euo pipefail

# ---- 颜色 ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR ]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }
header()  { echo -e "\n${BOLD}=== $* ===${NC}"; }

# ---- 默认配置 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_USER="ai"
FAST_IP_HEAD=""
FAST_IP_WORKER=""
NCCL_IF=""
NCCL_IF_WORKER=""
GLOO_IF=""
API_PORT=30000
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=4
GPU_MEM_UTIL=0.85

# ---- 参数解析 ----
WORKER_IP=""
IMAGE="ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603"
MODEL_PATH=""
NO_SYNC_MODEL=false
NO_SYNC_IMAGE=false
DRY_RUN=false
STOP_ONLY=false
STATUS_ONLY=false

while [ $# -gt 0 ]; do
    case "$1" in
        --no-sync-model)  NO_SYNC_MODEL=true ;;
        --no-sync-image)  NO_SYNC_IMAGE=true ;;
        --image)          shift; IMAGE="$1" ;;
        --dry-run)        DRY_RUN=true ;;
        --stop)           STOP_ONLY=true ;;
        --status)         STATUS_ONLY=true ;;
        --port)           shift; API_PORT="$1" ;;
        --max-len)        shift; MAX_MODEL_LEN="$1" ;;
        -h|--help)
            sed -n '2,/^# ==/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        -*)  die "未知选项: $1 (用 -h 查看帮助)" ;;
        *)
            if [ -z "$WORKER_IP" ]; then WORKER_IP="$1"
            
            elif [ -z "$MODEL_PATH" ]; then MODEL_PATH="$1"
            else die "多余参数: $1"; fi
            ;;
    esac
    shift
done

# ---- 状态模式 ----
if $STATUS_ONLY; then
    header "服务状态"
    docker ps --filter name=vllm-spark --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null
    echo ""
    if curl -sf --max-time 3 http://localhost:${API_PORT}/health &>/dev/null; then
        success "API 正常: http://$(hostname -I | awk '{print $1}'):${API_PORT}"
    else
        warn "API 未就绪"
    fi
    exit 0
fi

# ---- 停止模式 ----
if $STOP_ONLY; then
    header "停止服务"
    cd "$SCRIPT_DIR/runtime" && docker compose --profile head down 2>/dev/null || true
    [ -n "${WORKER_IP:-}" ] && ssh ${LOCAL_USER}@${WORKER_IP} "cd ~/dgx-spark-multinode/runtime && docker compose --profile worker down 2>/dev/null" || true
    # 兜底清理
    docker rm -f vllm-spark-head 2>/dev/null || true
    success "已停止"
    exit 0
fi

# ---- 参数校验 ----
[ -z "$WORKER_IP" ] && die "缺少工作节点 IP\n用法: bash $0 WORKER_IP MODEL_PATH [--image IMAGE]"
[ -z "$IMAGE" ] && die "缺少 Docker 镜像名\n用法: bash $0 WORKER_IP MODEL_PATH [--image IMAGE]"
[ -z "$MODEL_PATH" ] && die "缺少模型路径\n用法: bash $0 WORKER_IP MODEL_PATH [--image IMAGE]"

MODEL_PATH="$(eval echo "$MODEL_PATH")"
MODEL_PATH="$(cd "$MODEL_PATH" 2>/dev/null && pwd)" || die "模型路径不存在: $MODEL_PATH"
MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_CONTAINER_PATH="/models/${MODEL_NAME}"

# ---- 自动检测 ----
detect_local_mgmt_ip() {
    local prefix
    prefix=$(echo "$WORKER_IP" | grep -oP '^\d+\.\d+\.\d+\.')
    ip -4 addr show | grep -oP "${prefix}\d+" | head -1
}

detect_quantization() {
    if [ -f "$MODEL_PATH/hf_quant_config.json" ]; then
        local algo
        algo=$(python3 -c "import json; print(json.load(open('$MODEL_PATH/hf_quant_config.json')).get('quantization',{}).get('quant_algo',''))" 2>/dev/null)
        [ "$algo" = "NVFP4" ] && echo "modelopt_fp4" && return
    fi
    if [ -f "$MODEL_PATH/config.json" ]; then
        local qm
        qm=$(python3 -c "import json; print(json.load(open('$MODEL_PATH/config.json')).get('quantization_config',{}).get('quant_method',''))" 2>/dev/null)
        [ -n "$qm" ] && echo "$qm" && return
    fi
    echo ""
}

detect_cx7_interface() {
    # 在指定机器上找有 carrier 的 ConnectX-7 口, $1=空表示本机, 否则SSH
    local host="${1:-}"
    local cmd='for dev in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do [ -f /sys/class/net/$dev/carrier ] && [ "$(cat /sys/class/net/$dev/carrier 2>/dev/null)" = "1" ] && echo $dev && break; done'
    if [ -z "$host" ]; then
        bash -c "$cmd"
    else
        ssh -o ConnectTimeout=5 -o BatchMode=yes ${LOCAL_USER}@${host} "$cmd" 2>/dev/null
    fi
}

detect_fast_ip() {
    # 在指定接口上找 IP, $1=空表示本机
    local host="${1:-}" iface="$2"
    local cmd="ip -4 addr show $iface 2>/dev/null | grep -oP 'inet \K[^/]+' | head -1"
    if [ -z "$host" ]; then
        bash -c "$cmd"
    else
        ssh -o ConnectTimeout=5 -o BatchMode=yes ${LOCAL_USER}@${host} "$cmd" 2>/dev/null
    fi
}

LOCAL_MGMT_IP="$(detect_local_mgmt_ip)"
[ -z "$LOCAL_MGMT_IP" ] && die "无法检测本机管理网 IP"
QUANTIZATION="$(detect_quantization)"

# 检测本机高速网口和 IP
NCCL_IF="$(detect_cx7_interface)"
[ -z "$NCCL_IF" ] && die "本机未检测到有光缆的 ConnectX-7 口"
FAST_IP_HEAD="$(detect_fast_ip "" "$NCCL_IF")"
[ -z "$FAST_IP_HEAD" ] && die "本机高速网口 $NCCL_IF 未配置 IP"

# 检测 GLOO 接口 (管理网口)
GLOO_IF=$(ip -4 addr show | grep -B2 "$LOCAL_MGMT_IP" | grep -oP '^\d+: \K\S+(?=:)' | head -1)
[ -z "$GLOO_IF" ] && GLOO_IF="enP7s7"

# 构建 vLLM extra args
VLLM_EXTRA_ARGS="--enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
if echo "$MODEL_NAME" | grep -qi "qwen3"; then
    VLLM_EXTRA_ARGS="--enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
fi

# ============================================================
# 校验阶段
# ============================================================
header "预检校验"
PASS=0; FAIL=0

_check() {
    if eval "$2" &>/dev/null; then
        success "  $1"; PASS=$((PASS + 1))
    else
        error "  $1${3:+ — $3}"; FAIL=$((FAIL + 1))
    fi
}

# 本机校验
info "本机 (${LOCAL_MGMT_IP})"
_check "Docker 运行中" "docker info"
_check "本机镜像存在: $(echo $IMAGE | cut -d/ -f3)" "docker image inspect '$IMAGE'"
_check "模型路径存在: $MODEL_NAME" "test -d '$MODEL_PATH'"
_check "模型 config.json" "test -f '$MODEL_PATH/config.json'"
_check "模型 safetensors" "ls '$MODEL_PATH'/*.safetensors &>/dev/null"
_check "高速网口 $NCCL_IF" "ip link show $NCCL_IF | grep -q UP"
_check "高速网 IP ($FAST_IP_HEAD)" "ip addr show $NCCL_IF | grep -q $FAST_IP_HEAD"

# 工作节点校验
echo ""
info "工作节点 (${WORKER_IP})"
_check "SSH 连接" "ssh -o ConnectTimeout=5 -o BatchMode=yes ${LOCAL_USER}@${WORKER_IP} 'echo ok'"
_check "Docker 运行中" "ssh ${LOCAL_USER}@${WORKER_IP} 'docker info'"

# 检测工作节点光口和高速网 IP
NCCL_IF_WORKER="$(detect_cx7_interface "$WORKER_IP")"
if [ -z "$NCCL_IF_WORKER" ]; then
    error "  工作节点未检测到有光缆的 ConnectX-7 口"; FAIL=$((FAIL + 1))
else
    success "  工作节点光口: $NCCL_IF_WORKER"
    FAST_IP_WORKER="$(detect_fast_ip "$WORKER_IP" "$NCCL_IF_WORKER")"
    if [ -z "$FAST_IP_WORKER" ]; then
        error "  工作节点高速网口 $NCCL_IF_WORKER 未配置 IP"; FAIL=$((FAIL + 1))
    else
        success "  工作节点高速网 IP: $FAST_IP_WORKER"
    fi
fi
# 判断两台光口是否一致
if [ -n "$NCCL_IF" ] && [ -n "$NCCL_IF_WORKER" ]; then
    if [ "$NCCL_IF" = "$NCCL_IF_WORKER" ]; then
        NCCL_IF_COMPOSE="$NCCL_IF"
    else
        warn "  两台光口不同 ($NCCL_IF vs $NCCL_IF_WORKER), NCCL 自动检测"
        NCCL_IF_COMPOSE=""
    fi
fi
_check "高速网连通" "ping -c 1 -W 2 $FAST_IP_WORKER"

# 工作节点镜像 (仅警告)
if ssh ${LOCAL_USER}@${WORKER_IP} "docker image inspect '$IMAGE'" &>/dev/null; then
    success "  工作节点镜像存在"; PASS=$((PASS + 1))
else
    if $NO_SYNC_IMAGE; then
        error "  工作节点镜像不存在 (--no-sync-image 已设置)"; FAIL=$((FAIL + 1))
    else
        warn "  工作节点镜像不存在 (将自动传输)"
    fi
fi

# 工作节点模型
if ssh ${LOCAL_USER}@${WORKER_IP} "test -f '${MODEL_PATH}/config.json'" &>/dev/null; then
    success "  工作节点模型存在"; PASS=$((PASS + 1))
else
    if $NO_SYNC_MODEL; then
        error "  工作节点模型不存在 (--no-sync-model 已设置)"; FAIL=$((FAIL + 1))
    else
        warn "  工作节点模型不存在 (将自动同步)"
    fi
fi

# 模型大小 vs 可用内存
MODEL_SIZE_GB=$(du -s "$MODEL_PATH" | awk '{printf "%.0f", $1/1024/1024}')
info "  模型大小: ${MODEL_SIZE_GB}GB, 量化: ${QUANTIZATION:-none}"

echo ""
if [ "$FAIL" -gt 0 ]; then
    die "预检失败 ($FAIL 项)"
fi
success "预检通过 ($PASS 项)"

# ---- 配置摘要 ----
header "配置摘要"
info "本机 (HEAD):     ${LOCAL_MGMT_IP}"
info "工作节点:         ${WORKER_IP}"
info "镜像:             ${IMAGE}"
info "模型:             ${MODEL_NAME} (${MODEL_SIZE_GB}GB)"
info "量化:             ${QUANTIZATION:-auto}"
info "高速网:           ${FAST_IP_HEAD} <-> ${FAST_IP_WORKER} (${NCCL_IF})"
info "API:              http://${LOCAL_MGMT_IP}:${API_PORT}"

if $DRY_RUN; then
    info "[dry-run] 校验完成，不实际执行"
    exit 0
fi

# ============================================================
# 执行阶段
# ============================================================

# ==== Step 1: 停止旧服务 ====
header "Step 1/5: 清理旧容器"
docker rm -f vllm-spark-head vllm-single sglang-multinode-head 2>/dev/null || true
ssh ${LOCAL_USER}@${WORKER_IP} "docker rm -f vllm-spark-worker vllm-worker sglang-multinode-worker-1 2>/dev/null" || true
success "旧容器已清理"

# ==== Step 2: 同步镜像 ====
header "Step 2/5: 同步镜像到工作节点"
if $NO_SYNC_IMAGE; then
    info "跳过镜像同步 (--no-sync-image)"
elif ssh ${LOCAL_USER}@${WORKER_IP} "docker image inspect '$IMAGE'" &>/dev/null; then
    success "工作节点已有镜像"
else
    info "传输镜像到工作节点 (通过高速网)..."
    docker save "$IMAGE" | ssh ${LOCAL_USER}@${FAST_IP_WORKER} "docker load"
    success "镜像传输完成"
fi

# ==== Step 3: 同步模型 ====
header "Step 3/5: 同步模型到工作节点"
if $NO_SYNC_MODEL; then
    info "跳过模型同步 (--no-sync-model)"
elif ssh ${LOCAL_USER}@${WORKER_IP} "test -f '${MODEL_PATH}/config.json'" &>/dev/null; then
    success "工作节点模型已存在"
else
    SYNC_TARGET="$WORKER_IP"
    if ping -c 1 -W 1 "$FAST_IP_WORKER" &>/dev/null; then
        SYNC_TARGET="$FAST_IP_WORKER"
        info "使用高速网传输"
    fi
    ssh ${LOCAL_USER}@${WORKER_IP} "mkdir -p ${MODEL_PATH}"
    info "同步 ${MODEL_NAME} (${MODEL_SIZE_GB}GB)..."
    rsync -ah --info=progress2 "${MODEL_PATH}/" "${LOCAL_USER}@${SYNC_TARGET}:${MODEL_PATH}/"
    success "模型同步完成"
fi

# ==== Step 4: 生成配置并同步 ====
header "Step 4/5: 生成配置"

# 生成 .env
cat > "${SCRIPT_DIR}/runtime/.env" << ENVEOF
VLLM_IMAGE=${IMAGE}
MODEL_PATH=${MODEL_PATH}
MODEL_CONTAINER_PATH=${MODEL_CONTAINER_PATH}
SERVED_MODEL_NAME=${MODEL_NAME}
TP_SIZE=2
HEAD_ROCE_IP=${FAST_IP_HEAD}
WORKER_ROCE_IP=${FAST_IP_WORKER}
ROCE_IF_NAME=${NCCL_IF_COMPOSE:-${NCCL_IF}}
IB_HCA_NAME=roce$(echo "${NCCL_IF}" | sed 's/np[0-9]*$//' | sed 's/^en//')
RAY_PORT=6379
HOST_PORT=${API_PORT}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL}
MAX_NUM_BATCHED_TOKENS=${MAX_MODEL_LEN}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}
VLLM_MARLIN_USE_ATOMIC_ADD=0
VLLM_USE_FLASHINFER_MOE_FP4=0
VLLM_NVFP4_MOE_FORCE_MARLIN=0
GLOO_IF_NAME=${GLOO_IF}
ENVEOF

success "配置已生成"

# 同步到工作节点
info "同步配置到工作节点..."
ssh ${LOCAL_USER}@${WORKER_IP} "mkdir -p ~/dgx-spark-multinode/runtime/.cache/vllm"
rsync -ah "${SCRIPT_DIR}/runtime/" "${LOCAL_USER}@${WORKER_IP}:~/dgx-spark-multinode/runtime/"
success "配置已同步"

# ==== Step 5: 启动 ====
header "Step 5/5: 启动双节点"

# 启动 head
info "启动 HEAD (${LOCAL_MGMT_IP})..."
mkdir -p "${SCRIPT_DIR}/runtime/.cache/vllm"
cd "$SCRIPT_DIR/runtime" && docker compose --profile head up -d

# 等 Ray head 就绪
info "等待 Ray head 就绪..."
for i in $(seq 1 30); do
    if docker exec vllm-spark-head ray status &>/dev/null; then break; fi
    sleep 2
done

# 启动 worker
info "启动 WORKER (${WORKER_IP})..."
ssh ${LOCAL_USER}@${WORKER_IP} "cd ~/dgx-spark-multinode/runtime && docker compose --profile worker up -d"

# 等 2 个 GPU 加入
info "等待 Ray 集群就绪..."
for i in $(seq 1 60); do
    count=$(docker exec vllm-spark-head bash -c 'ray status 2>/dev/null | grep -c "node_"' 2>/dev/null || echo 0)
    [ "$count" -ge 2 ] && success "Ray 集群: $count 节点" && break
    sleep 5
done

# 等服务就绪
info "等待模型加载 (可能需要 10+ 分钟)..."
for i in $(seq 1 120); do
    if curl -sf --max-time 5 http://localhost:${API_PORT}/health &>/dev/null; then
        echo ""
        success "服务就绪!"
        break
    fi
    # 检查是否崩溃
    restarts=$(docker inspect -f '{{.RestartCount}}' vllm-spark-head 2>/dev/null || echo 0)
    if [ "$restarts" -gt 1 ]; then
        echo ""
        error "服务启动失败 (重启 ${restarts} 次)"
        info "查看日志: docker logs --tail 30 vllm-spark-head"
        exit 1
    fi
    echo -n "."
    sleep 10
done

# 测试推理
header "测试推理"
RESULT=$(curl -s --max-time 120 http://localhost:${API_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":50}" 2>&1)

if echo "$RESULT" | python3 -m json.tool &>/dev/null; then
    success "推理正常"
    echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['choices'][0]['message']; print(c.get('content') or c.get('reasoning','')[:200])"
else
    warn "推理测试失败: $RESULT"
fi

echo ""
header "部署完成"
info "API:       http://${LOCAL_MGMT_IP}:${API_PORT}"
info "健康检查:  curl http://${LOCAL_MGMT_IP}:${API_PORT}/health"
info "停止服务:  bash $0 --stop ${WORKER_IP}"
info "查看状态:  bash $0 --status"
info "查看日志:  docker logs -f --tail 50 vllm-spark-head"
