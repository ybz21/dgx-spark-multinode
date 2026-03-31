#!/bin/bash
# 共享函数库 - 被 deploy.sh 和子脚本 source

# ---- 颜色 ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }
header()  { echo -e "\n${BOLD}=== $* ===${NC}"; }

# ---- 常量 ----
VERSION="1.0.0"
CX7_IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"
DEPLOY_DIR="~/dgx-spark-multinode"

# ---- .env 加载 ----
load_env() {
    local env_file="${1:-.env}"
    local script_dir="${SCRIPT_DIR:-.}"
    local env_path="$script_dir/$env_file"

    if [ ! -f "$env_path" ]; then
        die "配置文件不存在: $env_path\n  请先复制模板: cp .env.example .env && vim .env"
    fi
    source "$env_path"

    # 校验必填变量
    [ ${#NODE_LIST[@]} -lt 2 ] && die "NODE_LIST 至少需要 2 个节点"
    [ -z "$MODEL_PATH" ]      && die "MODEL_PATH 未设置"
    [ -z "$SGLANG_IMAGE" ]    && die "SGLANG_IMAGE 未设置"
    [ -z "$SGLANG_PORT" ]     && die "SGLANG_PORT 未设置"
    [ -z "$TP_SIZE" ]         && die "TP_SIZE 未设置"

    NNODES=${#NODE_LIST[@]}
    if [ "$TP_SIZE" -ne "$NNODES" ]; then
        warn "TP_SIZE($TP_SIZE) != 节点数($NNODES)，确认这是你想要的配置"
    fi
}

# ---- 节点解析 ----
# parse_node <index> -> 设置 MGMT_IP FAST_IP USER HOSTNAME FAST_IFACE
parse_node() {
    local _iface
    IFS=',' read -r MGMT_IP FAST_IP USER HOSTNAME _iface <<< "${NODE_LIST[$1]}"
    FAST_IFACE="${_iface}"
}

# SSH 到指定节点
ssh_node() {
    local idx=$1; shift
    parse_node "$idx"
    ssh -o ConnectTimeout=5 -o BatchMode=yes "${USER}@${MGMT_IP}" "$@"
}

# ---- 光口检测 ----
# 远程自动检测有光缆的 ConnectX-7 口
# detect_iface <node_index> -> 设置 FAST_IFACE
detect_iface() {
    local idx=$1
    parse_node "$idx"
    if [ -n "$FAST_IFACE" ]; then return; fi

    local detected
    detected=$(ssh_node "$idx" "for dev in $CX7_IFACES; do [ -f /sys/class/net/\$dev/carrier ] && [ \$(cat /sys/class/net/\$dev/carrier 2>/dev/null) = 1 ] && echo \$dev && break; done" 2>/dev/null)
    if [ -z "$detected" ]; then
        warn "节点 $HOSTNAME ($MGMT_IP) 未检测到有光缆的口，使用默认 enp1s0f0np0"
        FAST_IFACE="enp1s0f0np0"
    else
        FAST_IFACE="$detected"
    fi
}

# 本地自动检测有光缆的口 (用于在目标机器上运行的脚本)
detect_iface_local() {
    for dev in $CX7_IFACES; do
        if [ -f "/sys/class/net/$dev/carrier" ] && [ "$(cat /sys/class/net/$dev/carrier 2>/dev/null)" = "1" ]; then
            echo "$dev"
            return
        fi
    done
}

# 从网口名推导 RDMA HCA: enp1s0f1np1 -> rocep1s0f1
iface_to_rdma() {
    echo "roce$(echo "$1" | sed 's/np[0-9]*$//' | sed 's/^en//')"
}
