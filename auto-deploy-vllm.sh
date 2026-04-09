#!/bin/bash
# 镜像拉取完成后，自动部署双节点 vLLM
# 被 auto-pull.sh 调用

set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

WORKER_IP="192.168.130.8"
WORKER_FAST_IP="10.0.0.2"
USER="ai"
IMAGE="ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603"
MODEL_PATH="/home/ai/models/Qwen3___5-122B-A10B-NVFP4"
SUDO_PASS="bladeai"

# 如果主源 tag 不存在，尝试备用
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603"; then
    IMAGE="ghcr.io/bjk110/vllm-spark:v019-ngc2603"
fi

log "使用镜像: $IMAGE"

# ===== Step 1: 传镜像到工作节点 =====
log "Step 1: 传输镜像到工作节点 ($WORKER_IP)..."
if ssh ${USER}@${WORKER_IP} "docker images --format '{{.Repository}}:{{.Tag}}' | grep -q 'vllm-spark:v019-ngc2603'"; then
    log "工作节点已有镜像，跳过"
else
    log "通过高速网传输镜像..."
    docker save "$IMAGE" | ssh ${USER}@${WORKER_FAST_IP} "docker load"
    log "镜像传输完成"
fi

# ===== Step 2: 确保模型在工作节点上 =====
log "Step 2: 检查工作节点模型..."
if ssh ${USER}@${WORKER_IP} "test -d ${MODEL_PATH}"; then
    log "工作节点模型已存在"
else
    log "同步模型到工作节点..."
    ssh ${USER}@${WORKER_IP} "mkdir -p ${MODEL_PATH}"
    rsync -ah --info=progress2 "${MODEL_PATH}/" "${USER}@${WORKER_FAST_IP}:${MODEL_PATH}/"
    log "模型同步完成"
fi

# ===== Step 3: 停旧容器 =====
log "Step 3: 清理旧容器..."
docker rm -f vllm-head vllm-single sglang-multinode-head 2>/dev/null || true
ssh ${USER}@${WORKER_IP} "docker rm -f vllm-worker vllm-test sglang-multinode-worker-1 2>/dev/null" || true
log "旧容器已清理"

# ===== Step 4: 检查 entrypoint =====
log "Step 4: 检查镜像 entrypoint..."
ENTRYPOINT=$(docker inspect "$IMAGE" --format '{{json .Config.Entrypoint}}')
log "Entrypoint: $ENTRYPOINT"

# ===== Step 5: 尝试双节点 TP=2 =====
log "Step 5: 启动双节点 TP=2 (Ray)..."

# Head
docker run -d --name vllm-head --gpus all --network host --ipc host --shm-size 32g \
  -e NCCL_SOCKET_IFNAME=enp \
  -e NCCL_IB_DISABLE=0 \
  -e GLOO_SOCKET_IFNAME=enP7s7 \
  -e VLLM_HOST_IP=192.168.130.16 \
  -e RAY_memory_monitor_refresh_ms=0 \
  -v ${MODEL_PATH}:/model:ro \
  --entrypoint bash \
  "$IMAGE" \
  -c '
ray start --head --port=6379 --node-ip-address=192.168.130.16 --num-gpus=1 --block &
sleep 20
echo "Waiting for 2 GPUs..."
for i in $(seq 1 120); do
  count=$(python3 -c "import ray; ray.init(address=\"auto\"); print(int(sum(n[\"Resources\"].get(\"GPU\",0) for n in ray.nodes() if n[\"Alive\"])))" 2>/dev/null | tail -1)
  [ "$count" = "2" ] && echo "Ray cluster: 2 GPUs ready" && break
  sleep 5
done
echo "Starting vLLM TP=2..."
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 --port 30000 \
  --quantization compressed-tensors \
  --trust-remote-code \
  --max-model-len 8192 \
  --distributed-executor-backend ray
'

# Worker
ssh ${USER}@${WORKER_IP} "docker run -d --name vllm-worker --gpus all --network host --ipc host --shm-size 32g \
  -e NCCL_SOCKET_IFNAME=enp \
  -e NCCL_IB_DISABLE=0 \
  -e GLOO_SOCKET_IFNAME=enP7s7 \
  -e VLLM_HOST_IP=192.168.130.8 \
  -e RAY_memory_monitor_refresh_ms=0 \
  -v ${MODEL_PATH}:/model:ro \
  --entrypoint bash \
  '$IMAGE' \
  -c 'ray start --address=192.168.130.16:6379 --node-ip-address=192.168.130.8 --num-gpus=1 --block'"

log "双节点容器已启动，等待服务就绪..."

# 等待最多 15 分钟
READY=false
for i in $(seq 1 90); do
    sleep 10
    if curl -sf --max-time 5 http://localhost:30000/health &>/dev/null; then
        READY=true
        break
    fi
    STATUS=$(docker inspect -f '{{.State.Status}}' vllm-head 2>/dev/null)
    if [ "$STATUS" != "running" ]; then
        log "双节点 head 容器退出，尝试单机模式..."
        break
    fi
done

if $READY; then
    log "===== 双节点 TP=2 部署成功! ====="
    log "API: http://192.168.130.16:30000"
    # 测试推理
    log "测试推理..."
    RESULT=$(curl -s --max-time 120 http://localhost:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"/model","messages":[{"role":"user","content":"你好"}],"max_tokens":50}')
    log "测试结果: $RESULT"
    exit 0
fi

# ===== Step 6: 双节点失败，回退到单机 TP=1 =====
log "Step 6: 双节点失败，回退到单机 TP=1..."
docker rm -f vllm-head 2>/dev/null || true
ssh ${USER}@${WORKER_IP} "docker rm -f vllm-worker 2>/dev/null" || true

docker run -d --name vllm-single --gpus all --network host --ipc host --shm-size 32g \
  -e RAY_memory_monitor_refresh_ms=0 \
  -v ${MODEL_PATH}:/model:ro \
  --entrypoint python3 \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
    --model /model \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 30000 \
    --quantization compressed-tensors \
    --trust-remote-code \
    --max-model-len 8192

log "单机容器已启动，等待就绪..."

for i in $(seq 1 90); do
    sleep 10
    if curl -sf --max-time 5 http://localhost:30000/health &>/dev/null; then
        log "===== 单机 TP=1 部署成功! ====="
        log "API: http://192.168.130.16:30000"
        RESULT=$(curl -s --max-time 120 http://localhost:30000/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{"model":"/model","messages":[{"role":"user","content":"你好"}],"max_tokens":50}')
        log "测试结果: $RESULT"
        exit 0
    fi
    STATUS=$(docker inspect -f '{{.State.Status}}' vllm-single 2>/dev/null)
    if [ "$STATUS" != "running" ]; then
        log "单机容器也退出了"
        docker logs --tail 20 vllm-single 2>&1
        break
    fi
done

log "===== 部署失败，请手动检查 ====="
exit 1
