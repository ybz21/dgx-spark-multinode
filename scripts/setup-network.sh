#!/bin/bash
# 配置 ConnectX-7 高速网络 (在目标机器上运行，需 sudo)
# 由 deploy.sh network 远程调用，一般不需要手动执行

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"
source "$SCRIPT_DIR/.env" 2>/dev/null || die "无法加载 .env"

NODE_NUM=${1:-""}
[ -z "$NODE_NUM" ] && die "用法: sudo bash $0 <节点序号> (从1开始)"

NODE_IDX=$((NODE_NUM - 1))
[ "$NODE_IDX" -ge "${#NODE_LIST[@]}" ] && die "节点 $NODE_NUM 不存在 (共 ${#NODE_LIST[@]} 个)"

IFS=',' read -r MGMT_IP FAST_IP USER HOSTNAME IFACE <<< "${NODE_LIST[$NODE_IDX]}"

# 自动检测光口
if [ -z "$IFACE" ]; then
    info "自动检测有光缆的 ConnectX-7 口..."
    IFACE=$(detect_iface_local)
    if [ -z "$IFACE" ]; then
        error "未检测到有光缆的口，各口状态:"
        for dev in $CX7_IFACES; do
            carrier=$(cat /sys/class/net/$dev/carrier 2>/dev/null || echo "?")
            echo "  $dev: carrier=$carrier"
        done
        exit 1
    fi
    success "检测到: $IFACE"
fi

# 检查 root
[ "$EUID" -ne 0 ] && die "需要 root 权限，请用 sudo 运行"

# 检查 nmcli
command -v nmcli &>/dev/null || die "nmcli 未安装，请安装 NetworkManager"

# 检查接口存在
ip link show "$IFACE" &>/dev/null || die "接口 $IFACE 不存在"

header "配置节点 $NODE_NUM ($HOSTNAME)"
info "接口: $IFACE | IP: $FAST_IP/$FAST_SUBNET | MTU: $FAST_MTU"

# 检查是否已配置
EXISTING_IP=$(ip -4 addr show "$IFACE" 2>/dev/null | grep -oP 'inet \K[^/]+')
if [ "$EXISTING_IP" = "$FAST_IP" ]; then
    success "已配置，跳过 (IP: $FAST_IP, 接口: $IFACE)"
    exit 0
fi

nmcli con delete cx7-multinode 2>/dev/null || true

nmcli con add \
    type ethernet \
    con-name cx7-multinode \
    ifname "$IFACE" \
    ipv4.method manual \
    ipv4.addresses "${FAST_IP}/${FAST_SUBNET}" \
    ipv6.method disabled \
    ethernet.mtu "$FAST_MTU" \
    connection.autoconnect yes \
    > /dev/null

if nmcli con up cx7-multinode &>/dev/null; then
    # 验证
    ACTUAL_IP=$(ip -4 addr show "$IFACE" 2>/dev/null | grep -oP 'inet \K[^/]+')
    ACTUAL_MTU=$(cat /sys/class/net/$IFACE/mtu 2>/dev/null)
    if [ "$ACTUAL_IP" = "$FAST_IP" ]; then
        success "配置完成: $IFACE = $ACTUAL_IP (MTU $ACTUAL_MTU)"
    else
        warn "配置已应用但 IP 验证失败 (期望 $FAST_IP, 实际 $ACTUAL_IP)"
    fi
else
    warn "连接已配置，等待光缆插入后自动激活"
fi
