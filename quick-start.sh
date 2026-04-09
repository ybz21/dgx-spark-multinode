#!/bin/bash
# ============================================================
# DGX Spark 双节点快速部署
# 以本机为主节点，自动同步模型和配置到工作节点，然后启动推理
#
# 用法:
#   bash quick-start.sh <工作节点IP> <模型路径>
#   bash quick-start.sh 192.168.130.8 ~/models/Qwen3.5-35B-A3B-NVFP4-txn545
#
# 选项:
#   --no-sync    跳过模型同步 (模型已在工作节点上)
#   --dry-run    仅显示配置，不实际执行
#   --stop       停止当前运行的服务
# ============================================================

set -euo pipefail

# ---- 颜色 ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }
header()  { echo -e "\n${BOLD}=== $* ===${NC}"; }

# ---- 固定配置 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_USER="ai"
LOCAL_HOSTNAME="$(hostname)"
SGLANG_IMAGE="sglang-dev-cu13-accel:latest"
SGLANG_PORT=30000
NCCL_INIT_PORT=29500
ATTENTION_BACKEND="triton"
MEM_FRACTION=0.8
FAST_SUBNET=24
FAST_MTU=9000
FAST_IP_HEAD="10.0.0.1"
FAST_IP_WORKER="10.0.0.2"
CUSTOM_SCRIPTS_DIR="/home/ai/lm_scripts/services"
CUSTOM_FILES="sglang_entrypoint.sh qwen3_nothink_reasoning_detector.py qwen3_coder_nothink_detector.py patch_parser.py"
CUSTOM_ENTRYPOINT="sglang_entrypoint.sh"

# ---- 参数解析 ----
WORKER_IP=""
MODEL_PATH=""
NO_SYNC=false
DRY_RUN=false
STOP_ONLY=false

while [ $# -gt 0 ]; do
    case "$1" in
        --no-sync)  NO_SYNC=true ;;
        --dry-run)  DRY_RUN=true ;;
        --stop)     STOP_ONLY=true ;;
        -h|--help)
            echo "用法: bash $0 WORKER_IP MODEL_PATH [--no-sync] [--dry-run] [--stop]"
            echo ""
            echo "参数:"
            echo "  WORKER_IP    另一台 DGX Spark 的管理网 IP (如 192.168.130.8)"
            echo "  MODEL_PATH   本机模型目录 (如 ~/models/Qwen3.5-35B-A3B-NVFP4-txn545)"
            echo ""
            echo "选项:"
            echo "  --no-sync    跳过模型同步 (模型已在工作节点上)"
            echo "  --dry-run    仅显示配置，不实际执行"
            echo "  --stop       停止当前运行的服务"
            exit 0
            ;;
        -*)  die "未知选项: $1" ;;
        *)
            if [ -z "$WORKER_IP" ]; then
                WORKER_IP="$1"
            elif [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            else
                die "多余的参数: $1"
            fi
            ;;
    esac
    shift
done

# ---- 停止模式 ----
if $STOP_ONLY; then
    header "停止服务"
    cd "$SCRIPT_DIR" && bash deploy.sh stop -y
    exit 0
fi

# ---- 参数校验 ----
[ -z "$WORKER_IP" ] && die "缺少工作节点 IP\n用法: bash $0 WORKER_IP MODEL_PATH"
[ -z "$MODEL_PATH" ] && die "缺少模型路径\n用法: bash $0 WORKER_IP MODEL_PATH"

# 展开 ~ 和相对路径
MODEL_PATH="$(eval echo "$MODEL_PATH")"
MODEL_PATH="$(cd "$MODEL_PATH" 2>/dev/null && pwd)" || die "模型路径不存在: $MODEL_PATH"

# ---- 自动检测本机管理网 IP ----
detect_local_mgmt_ip() {
    local ip
    # 优先取和工作节点同网段的 IP
    local worker_prefix
    worker_prefix=$(echo "$WORKER_IP" | grep -oP '^\d+\.\d+\.\d+\.')
    ip=$(ip -4 addr show | grep -oP "${worker_prefix}\d+" | head -1)
    [ -n "$ip" ] && echo "$ip" && return
    # fallback: 取默认路由的源 IP
    ip=$(ip route get "$WORKER_IP" 2>/dev/null | grep -oP 'src \K\S+')
    echo "$ip"
}

LOCAL_MGMT_IP="$(detect_local_mgmt_ip)"
[ -z "$LOCAL_MGMT_IP" ] && die "无法检测本机管理网 IP"

# ---- 自动检测量化方式 ----
detect_quantization() {
    local config="$MODEL_PATH/hf_quant_config.json"
    if [ -f "$config" ]; then
        local algo
        algo=$(python3 -c "import json; print(json.load(open('$config')).get('quantization',{}).get('quant_algo',''))" 2>/dev/null)
        case "$algo" in
            NVFP4) echo "modelopt_fp4"; return ;;
        esac
    fi
    # 从 config.json 检测
    if [ -f "$MODEL_PATH/config.json" ]; then
        local qmethod
        qmethod=$(python3 -c "import json; c=json.load(open('$MODEL_PATH/config.json')); print(c.get('quantization_config',{}).get('quant_method',''))" 2>/dev/null)
        case "$qmethod" in
            compressed-tensors) echo "compressed-tensors"; return ;;
            modelopt)           echo "modelopt_fp4"; return ;;
        esac
    fi
    echo ""
}

QUANTIZATION="$(detect_quantization)"

# ---- 自动检测工作节点主机名 ----
detect_worker_hostname() {
    ssh -o ConnectTimeout=5 -o BatchMode=yes "${LOCAL_USER}@${WORKER_IP}" "hostname" 2>/dev/null || echo "worker"
}

# ---- 显示配置摘要 ----
MODEL_NAME="$(basename "$MODEL_PATH")"
header "配置摘要"
info "本机 (HEAD):   ${LOCAL_MGMT_IP} (${LOCAL_HOSTNAME})"
info "工作节点:       ${WORKER_IP}"
info "模型:           ${MODEL_NAME}"
info "模型路径:       ${MODEL_PATH}"
info "量化方式:       ${QUANTIZATION:-无}"
info "SGLang 镜像:    ${SGLANG_IMAGE}"
info "高速网:         ${FAST_IP_HEAD} <-> ${FAST_IP_WORKER}"
info "API 端口:       ${SGLANG_PORT}"

if $DRY_RUN; then
    info "[dry-run] 仅显示配置，退出"
    exit 0
fi

# ==== Step 1: SSH 连通性 ====
header "Step 1/6: 检查 SSH 连接"
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${LOCAL_USER}@${WORKER_IP}" "echo ok" &>/dev/null; then
    success "SSH 连接正常"
else
    die "无法 SSH 到 ${LOCAL_USER}@${WORKER_IP}\n请先配置免密登录: ssh-copy-id ${LOCAL_USER}@${WORKER_IP}"
fi

WORKER_HOSTNAME="$(detect_worker_hostname)"
info "工作节点主机名: ${WORKER_HOSTNAME}"

# ==== Step 2: 停止旧服务 ====
header "Step 2/6: 停止旧服务"
docker rm -f sglang-multinode-head sglang-multinode-worker-1 2>/dev/null || true
ssh "${LOCAL_USER}@${WORKER_IP}" "docker rm -f sglang-multinode-head sglang-multinode-worker-1 2>/dev/null" || true
success "旧容器已清理"

# ==== Step 3: 同步模型 ====
header "Step 3/6: 同步模型到工作节点"
if $NO_SYNC; then
    info "跳过模型同步 (--no-sync)"
    if ! ssh "${LOCAL_USER}@${WORKER_IP}" "test -d ${MODEL_PATH}"; then
        die "工作节点上模型不存在: ${MODEL_PATH}\n去掉 --no-sync 以自动同步"
    fi
    success "工作节点模型已存在"
else
    SYNC_TARGET="${WORKER_IP}"
    if ping -c 1 -W 1 "$FAST_IP_WORKER" &>/dev/null; then
        SYNC_TARGET="$FAST_IP_WORKER"
        info "使用高速网 (${FAST_IP_WORKER}) 传输"
    else
        info "使用管理网 (${WORKER_IP}) 传输"
    fi
    ssh "${LOCAL_USER}@${WORKER_IP}" "mkdir -p ${MODEL_PATH}"
    info "开始同步 ${MODEL_NAME} ..."
    rsync -ah --info=progress2 "${MODEL_PATH}/" "${LOCAL_USER}@${SYNC_TARGET}:${MODEL_PATH}/"
    success "模型同步完成"
fi

# ==== Step 4: 生成配置 ====
header "Step 4/6: 生成配置"

# 检测并同步自定义脚本
HAS_CUSTOM=false
CUSTOM_BLOCK=""
if [ -d "$CUSTOM_SCRIPTS_DIR" ]; then
    local_ok=true
    for f in $CUSTOM_FILES; do
        [ -f "${CUSTOM_SCRIPTS_DIR}/${f}" ] || { local_ok=false; break; }
    done
    if $local_ok; then
        # 确保工作节点上也有
        remote_ok=true
        for f in $CUSTOM_FILES; do
            ssh "${LOCAL_USER}@${WORKER_IP}" "test -f ${CUSTOM_SCRIPTS_DIR}/${f}" 2>/dev/null || { remote_ok=false; break; }
        done
        if ! $remote_ok; then
            info "同步自定义脚本到工作节点..."
            ssh "${LOCAL_USER}@${WORKER_IP}" "mkdir -p ${CUSTOM_SCRIPTS_DIR}"
            for f in $CUSTOM_FILES; do
                scp -q "${CUSTOM_SCRIPTS_DIR}/${f}" "${LOCAL_USER}@${WORKER_IP}:${CUSTOM_SCRIPTS_DIR}/${f}"
            done
        fi
        HAS_CUSTOM=true
        CUSTOM_BLOCK="CUSTOM_SCRIPTS_DIR=${CUSTOM_SCRIPTS_DIR}
CUSTOM_FILES=\"${CUSTOM_FILES}\"
CUSTOM_ENTRYPOINT=${CUSTOM_ENTRYPOINT}"
    fi
fi

# 自动选择 tool-call parser
EXTRA_ARGS="--trust-remote-code"
if echo "$MODEL_NAME" | grep -qi "qwen3"; then
    if $HAS_CUSTOM; then
        EXTRA_ARGS="--trust-remote-code --tool-call-parser qwen3_coder_nothink --reasoning-parser qwen3"
    else
        EXTRA_ARGS="--trust-remote-code --tool-call-parser qwen3_coder --reasoning-parser qwen3"
    fi
fi

# 写入 .env
cat > "${SCRIPT_DIR}/.env" << ENVEOF
NODE_LIST=(
    "${LOCAL_MGMT_IP},${FAST_IP_HEAD},${LOCAL_USER},${LOCAL_HOSTNAME}"
    "${WORKER_IP},${FAST_IP_WORKER},${LOCAL_USER},${WORKER_HOSTNAME}"
)

FAST_SUBNET=${FAST_SUBNET}
FAST_MTU=${FAST_MTU}

MODEL_PATH=${MODEL_PATH}

SGLANG_IMAGE=${SGLANG_IMAGE}
SGLANG_PORT=${SGLANG_PORT}
NCCL_INIT_PORT=${NCCL_INIT_PORT}
TP_SIZE=2
ATTENTION_BACKEND=${ATTENTION_BACKEND}
MEM_FRACTION=${MEM_FRACTION}
QUANTIZATION=${QUANTIZATION}
EXTRA_ARGS="${EXTRA_ARGS}"
${CUSTOM_BLOCK}
ENVEOF

success "配置已生成"

# ==== Step 5: 同步部署脚本 ====
header "Step 5/6: 同步部署脚本"
cd "$SCRIPT_DIR" && bash deploy.sh sync

# ==== Step 6: 启动 ====
header "Step 6/6: 启动推理"
cd "$SCRIPT_DIR" && bash deploy.sh start

echo ""
header "部署完成"
info "API 地址:  http://${LOCAL_MGMT_IP}:${SGLANG_PORT}"
info "测试推理:  bash deploy.sh test"
info "查看日志:  bash deploy.sh logs 1 -f"
info "停止服务:  bash $(basename "$0") --stop"
