#!/bin/bash
# 测试高速网络连通性
# 用法: bash test-connection.sh <本机节点号> [目标节点号]
# 示例: bash test-connection.sh 1 2

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

# ConnectX-7 网口名列表
CX7_IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

LOCAL_NUM=${1:?"用法: bash test-connection.sh <本机节点号> [目标节点号]"}
TARGET_NUM=${2:-$((LOCAL_NUM == 1 ? 2 : 1))}

LOCAL_IDX=$((LOCAL_NUM - 1))
TARGET_IDX=$((TARGET_NUM - 1))

IFS=',' read -r _ LOCAL_IP _ _ LOCAL_IFACE <<< "${NODE_LIST[$LOCAL_IDX]}"
IFS=',' read -r _ TARGET_IP _ TARGET_HOST _ <<< "${NODE_LIST[$TARGET_IDX]}"

# 自动检测接口
if [ -z "$LOCAL_IFACE" ]; then
    for dev in $CX7_IFACES; do
        if [ -f "/sys/class/net/$dev/carrier" ] && [ "$(cat /sys/class/net/$dev/carrier 2>/dev/null)" = "1" ]; then
            LOCAL_IFACE="$dev"
            break
        fi
    done
fi
LOCAL_IFACE="${LOCAL_IFACE:-enp1s0f0np0}"
RDMA_HCA="roce$(echo "$LOCAL_IFACE" | sed 's/np[0-9]*$//' | sed 's/^en//')"

echo "=== 测试: 节点$LOCAL_NUM ($LOCAL_IP, $LOCAL_IFACE) -> 节点$TARGET_NUM ($TARGET_IP) ==="
echo ""

echo "--- 接口状态 ---"
ip -br addr show "$LOCAL_IFACE" 2>/dev/null || echo "接口不存在"
ethtool "$LOCAL_IFACE" 2>/dev/null | grep -E "Speed|Link" || true

echo ""
echo "--- Ping ---"
ping -c 3 -I "$LOCAL_IP" "$TARGET_IP" || echo "FAIL"

echo ""
echo "--- Jumbo Frame (MTU $FAST_MTU) ---"
ping -c 1 -M do -s $((FAST_MTU - 28)) -I "$LOCAL_IP" "$TARGET_IP" \
    && echo "OK" || echo "FAIL"

echo ""
echo "--- RDMA ---"
ibv_devinfo -d "$RDMA_HCA" 2>/dev/null | grep -E "hca_id|state|link_layer" || echo "无 RDMA"

echo ""
echo "--- 手动带宽测试 ---"
echo "iperf3:  对端: iperf3 -s -B $TARGET_IP"
echo "         本机: iperf3 -c $TARGET_IP -B $LOCAL_IP -t 10"
echo "RDMA:    对端: ib_write_bw -d $RDMA_HCA --report_gbits"
echo "         本机: ib_write_bw -d $RDMA_HCA $TARGET_IP --report_gbits"
