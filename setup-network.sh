#!/bin/bash
# 配置 ConnectX-7 高速网络 (需 sudo)
# 用法: sudo bash setup-network.sh <节点序号>  (从 1 开始)
# 自动检测有光缆的 ConnectX-7 口，或使用 NODE_LIST 中指定的接口

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

# ConnectX-7 网口名列表 (DGX Spark 上的 4 个口)
CX7_IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

NODE_NUM=${1:?"用法: sudo bash setup-network.sh <节点序号> (从1开始)"}
NODE_IDX=$((NODE_NUM - 1))

if [ "$NODE_IDX" -ge "${#NODE_LIST[@]}" ]; then
    echo "错误: 节点 $NODE_NUM 不存在 (共 ${#NODE_LIST[@]} 个节点)"
    exit 1
fi

IFS=',' read -r MGMT_IP FAST_IP USER HOSTNAME IFACE <<< "${NODE_LIST[$NODE_IDX]}"

# 自动检测光口
if [ -z "$IFACE" ]; then
    echo "未指定接口，自动检测有光缆的 ConnectX-7 口..."
    for dev in $CX7_IFACES; do
        if [ -f "/sys/class/net/$dev/carrier" ] && [ "$(cat /sys/class/net/$dev/carrier 2>/dev/null)" = "1" ]; then
            IFACE="$dev"
            echo "  检测到: $dev (carrier=1)"
            break
        fi
    done
    if [ -z "$IFACE" ]; then
        echo "错误: 未检测到有光缆的 ConnectX-7 口"
        echo "可用接口及状态:"
        for dev in $CX7_IFACES; do
            carrier=$(cat /sys/class/net/$dev/carrier 2>/dev/null || echo "?")
            echo "  $dev: carrier=$carrier"
        done
        exit 1
    fi
fi

echo "=== 配置节点 $NODE_NUM ($HOSTNAME) ==="
echo "接口: $IFACE | IP: $FAST_IP/$FAST_SUBNET | MTU: $FAST_MTU"

if ! ip link show "$IFACE" &>/dev/null; then
    echo "错误: 接口 $IFACE 不存在。可用接口:"
    ip -br link show | grep -v "lo\|docker\|veth\|br-\|tailscale"
    exit 1
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
    connection.autoconnect yes

nmcli con up cx7-multinode 2>/dev/null \
    && echo "连接已激活" \
    || echo "连接已配置，插上光缆后自动激活"
