#!/bin/bash
# 自动拉取 Docker 镜像，失败自动重试，主备源切换
# 用法: nohup bash auto-pull.sh > ~/auto-pull.log 2>&1 &

set -o pipefail

PRIMARY="ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603"
BACKUP="ghcr.io/bjk110/vllm-spark:v019-ngc2603"
TARGET_TAG="vllm-spark:v019-ngc2603"
MAX_RETRIES=50
RETRY_DELAY=30
FAIL_THRESHOLD=10  # 连续失败几次后切换源

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 检查镜像是否已存在
check_image() {
    docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "$TARGET_TAG"
}

if check_image; then
    log "镜像已存在，无需拉取"
    exit 0
fi

CURRENT_SOURCE="$PRIMARY"
CONSECUTIVE_FAILS=0
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "=== 第 ${ATTEMPT}/${MAX_RETRIES} 次尝试 === 源: $CURRENT_SOURCE"

    # 启动 docker pull 并监控
    docker pull "$CURRENT_SOURCE" &
    PULL_PID=$!

    # 监控 pull 进程：每30秒检查是否还活着，最长等30分钟
    WAIT=0
    WAIT_MAX=1800
    while kill -0 $PULL_PID 2>/dev/null; do
        sleep 30
        WAIT=$((WAIT + 30))
        if [ $WAIT -ge $WAIT_MAX ]; then
            log "超时 ${WAIT_MAX}s，kill docker pull (PID=$PULL_PID)"
            kill $PULL_PID 2>/dev/null
            sleep 5
            kill -9 $PULL_PID 2>/dev/null
            break
        fi
        # 打印进度
        DL=$(docker pull "$CURRENT_SOURCE" --quiet 2>/dev/null | wc -c || true)
        log "等待中... ${WAIT}s (PID=$PULL_PID)"
    done

    wait $PULL_PID 2>/dev/null
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ] && check_image; then
        log "拉取成功! 源: $CURRENT_SOURCE"
        # 如果是从备用源拉的，tag 成统一名称
        if [ "$CURRENT_SOURCE" != "$PRIMARY" ]; then
            docker tag "$CURRENT_SOURCE" "$PRIMARY" 2>/dev/null || true
        fi
        # ===== 拉取成功后自动部署 =====
        log "开始自动部署双节点..."
        bash ~/dgx-spark-multinode/auto-deploy-vllm.sh
        exit 0
    fi

    CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))
    log "失败 (exit=$EXIT_CODE, 连续失败=$CONSECUTIVE_FAILS)"

    # 连续失败超过阈值，切换源
    if [ $CONSECUTIVE_FAILS -ge $FAIL_THRESHOLD ]; then
        if [ "$CURRENT_SOURCE" = "$PRIMARY" ]; then
            log "切换到备用源: $BACKUP"
            CURRENT_SOURCE="$BACKUP"
        else
            log "切换回主源: $PRIMARY"
            CURRENT_SOURCE="$PRIMARY"
        fi
        CONSECUTIVE_FAILS=0
    fi

    log "等待 ${RETRY_DELAY}s 后重试..."
    sleep $RETRY_DELAY
done

log "达到最大重试次数 $MAX_RETRIES，放弃"
exit 1
