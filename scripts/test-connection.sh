#!/bin/bash
# 测试高速网络连通性 (在目标机器上运行)
# 由 deploy.sh test-net 远程调用

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"
source "$SCRIPT_DIR/.env" 2>/dev/null || die "无法加载 .env"

LOCAL_NUM=${1:-""}
TARGET_NUM=${2:-""}
[ -z "$LOCAL_NUM" ] && die "用法: bash $0 <本机节点号> [目标节点号]"
[ -z "$TARGET_NUM" ] && TARGET_NUM=$((LOCAL_NUM == 1 ? 2 : 1))

LOCAL_IDX=$((LOCAL_NUM - 1))
TARGET_IDX=$((TARGET_NUM - 1))
[ "$LOCAL_IDX" -ge "${#NODE_LIST[@]}" ] && die "节点 $LOCAL_NUM 不存在"
[ "$TARGET_IDX" -ge "${#NODE_LIST[@]}" ] && die "节点 $TARGET_NUM 不存在"

IFS=',' read -r _ LOCAL_IP _ LOCAL_HOST LOCAL_IFACE <<< "${NODE_LIST[$LOCAL_IDX]}"
IFS=',' read -r _ TARGET_IP _ TARGET_HOST _ <<< "${NODE_LIST[$TARGET_IDX]}"

# 自动检测接口
[ -z "$LOCAL_IFACE" ] && LOCAL_IFACE=$(detect_iface_local)
LOCAL_IFACE="${LOCAL_IFACE:-enp1s0f0np0}"
RDMA_HCA="$(iface_to_rdma "$LOCAL_IFACE")"

PASS=0; FAIL=0; TOTAL=0
check() {
    TOTAL=$((TOTAL + 1))
    if eval "$2" &>/dev/null; then
        success "$1"; PASS=$((PASS + 1))
    else
        error "$1"; FAIL=$((FAIL + 1))
    fi
}

header "连接测试: $LOCAL_HOST ($LOCAL_IP) -> $TARGET_HOST ($TARGET_IP)"
info "接口: $LOCAL_IFACE | RDMA: $RDMA_HCA"
echo ""

# 1. 接口状态
echo -e "${BOLD}--- 接口状态 ---${NC}"
ip -br addr show "$LOCAL_IFACE" 2>/dev/null || warn "接口不存在"
ethtool "$LOCAL_IFACE" 2>/dev/null | grep -E "Speed|Link" || true
echo ""

# 2. 测试项
echo -e "${BOLD}--- 测试项 ---${NC}"
check "Ping 连通性" "ping -c 1 -W 3 -I $LOCAL_IP $TARGET_IP"
check "Jumbo Frame (MTU ${FAST_MTU:-9000})" "ping -c 1 -W 3 -M do -s $((${FAST_MTU:-9000} - 28)) -I $LOCAL_IP $TARGET_IP"

if command -v ibv_devinfo &>/dev/null; then
    check "RDMA 设备 ($RDMA_HCA)" "ibv_devinfo -d $RDMA_HCA"
else
    warn "ibv_devinfo 未安装，跳过 RDMA 检查"; TOTAL=$((TOTAL + 1)); FAIL=$((FAIL + 1))
fi

# 3. 汇总
echo ""
echo -e "${BOLD}--- 结果 ---${NC}"
if [ "$FAIL" -eq 0 ]; then
    success "全部通过 ($PASS/$TOTAL)"
else
    error "失败 $FAIL/$TOTAL"
fi

echo ""
echo -e "${BOLD}--- 手动带宽测试 ---${NC}"
echo "  iperf3:  对端: iperf3 -s -B $TARGET_IP"
echo "           本机: iperf3 -c $TARGET_IP -B $LOCAL_IP -t 10"
echo "  RDMA:    对端: ib_write_bw -d $RDMA_HCA --report_gbits"
echo "           本机: ib_write_bw -d $RDMA_HCA $TARGET_IP --report_gbits"

exit $FAIL
