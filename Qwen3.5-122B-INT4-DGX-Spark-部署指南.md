# Qwen3.5-122B-A10B INT4 AutoRound — DGX Spark 单机部署指南

## 概述

使用 Intel/Qwen3.5-122B-A10B-int4-AutoRound 模型，基于社区方案 [albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) 构建 SM121 专用 vLLM 镜像，配合 FlashInfer 注意力后端 + MTP-2 投机解码，**单台 DGX Spark 实测生成速度 46 tok/s（峰值 50.8 tok/s）**，相比 NVFP4 双节点方案的 16.6 tok/s 提升约 3 倍。

## 性能对比

| 方案 | 速度 | 节点数 | 备注 |
|---|---|---|---|
| NVFP4 双节点 (通用 vLLM) | 16.6 tok/s | 2 | 缺少 SM121 CUTLASS kernels |
| INT4 AutoRound (通用 vLLM) | 17 tok/s | 1 | 无 MTP，通用 FlashAttn |
| **INT4 AutoRound (SM121 + MTP-2)** | **46 tok/s** | **1** | 本方案 |

## 关键配置

- **模型**: Intel/Qwen3.5-122B-A10B-int4-AutoRound（GPTQ INT4，72GB）
- **镜像**: `vllm-qwen35-v2`（基于 eugr/spark-vllm-docker SM121 编译 + albond patch）
- **注意力后端**: FlashInfer（SM121 专用编译）
- **投机解码**: MTP-2（Multi-Token Prediction，num_speculative_tokens=2）
- **MTP-2 统计**: 平均接受长度 2.4-2.9 tokens，接受率 73-93%
- **API**: OpenAI 兼容接口，支持 reasoning（思考链）+ tool calling + vision（多模态图片理解）

---

## 内存分析

### 硬件基础

- GPU: NVIDIA GB10（DGX Spark）
- 统一内存: 128 GiB（GPU + CPU 共享）
- gpu-memory-utilization: 0.90 → 可用约 115.2 GiB

### 一、权重内存（静态，固定占用）

| 项目 | 占用 |
|---|---|
| 模型权重（INT4 量化，72GB 磁盘） | 67.35 GiB（vLLM 实测） |
| MTP 投机解码权重（model_extra_tensors, 4.8GB） | 包含在上面 |
| CUDA graph | 0.55 GiB |
| 框架/运行时开销（PyTorch, NCCL 等） | ~2 GiB |
| 桌面环境（Xorg + GNOME） | ~0.3 GiB |
| **权重总计** | **~70.2 GiB** |

### 二、动态内存（KV Cache）

可用 KV cache 内存 = 115.2 - 70.2 = ~45 GiB

vLLM 实际报告: **Available KV cache memory: 33.02 GiB**（扣除 profiling 预留和激活值开销后）

实际 KV cache 容量: **331,584 tokens**

### 三、不同上下文长度支持

| max-model-len | 单请求占用 | 最大并发 | 适用场景 |
|---|---|---|---|
| 32K（当前配置） | ~3.2 GiB | 25 路 | 多用户对话，高并发 |
| 65K | ~6.4 GiB | 12 路 | 长文档分析 |
| 131K（128K） | ~12.8 GiB | 5 路 | 长上下文推理 |
| 262K（256K） | ~25.6 GiB | 2 路 | 极长上下文，低并发 |
| 331K（理论最大） | ~33 GiB | 1 路 | 单请求极限 |

---

## 部署步骤

### 前置条件

- DGX Spark（NVIDIA GB10, 128GB 统一内存）
- 模型已下载: `~/models/Qwen3.5-122B-A10B-int4-AutoRound-Intel`
- Docker 镜像已构建: `vllm-sm121`, `vllm-qwen35-v2`

### 镜像构建（仅首次）

```bash
# 1. 克隆仓库
git clone https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4.git
cd DGX_Spark_Qwen3.5-122B-A10B-AR-INT4

# 2. 克隆 spark-vllm-docker 并切到指定 commit
git clone https://github.com/eugr/spark-vllm-docker.git spark-vllm-docker
cd spark-vllm-docker
git checkout --force 49d6d9fefd7cd05e63af8b28e4b514e9d30d249f

# 3. 确保本地有 CUDA 基础镜像
docker pull nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04
docker tag nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 nvidia/cuda:13.2.0-devel-ubuntu24.04

# 4. 构建 vllm-sm121 基础镜像（约 30-60 分钟）
./build-and-copy.sh -t vllm-sm121 --vllm-ref v0.19.0 --tf5

# 5. 构建 vllm-qwen35-v2 最终镜像（几秒）
cd ..
docker build -t vllm-qwen35-v2 -f docker/Dockerfile.v2 .
```

### 启动服务

```bash
# 清理旧容器
docker rm -f vllm-qwen35 2>/dev/null || true

# 启动（单机，MTP-2 + FlashInfer）
docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v /home/ai/models:/models \
  vllm-qwen35-v2 \
  serve /models/Qwen3.5-122B-A10B-int4-AutoRound-Intel \
  --served-model-name qwen3.5-122b-int4 \
  --port 30000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

模型加载约 10 分钟，等待健康检查通过:

```bash
# 健康检查
curl http://localhost:30000/health

# 查看日志
docker logs -f --tail 50 vllm-qwen35
```

### 调整上下文长度

修改 `--max-model-len` 参数即可，例如 256K 上下文:

```bash
--max-model-len 262144
```

注意：上下文越大，并发能力越低。

---

## 测试示例

### 文本对话

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b-int4",
    "messages": [{"role": "user", "content": "1+1等于几？"}],
    "max_tokens": 1000
  }'
```

### 图片理解（Vision）

```python
import json, base64, urllib.request

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

data = json.dumps({
    "model": "qwen3.5-122b-int4",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": "描述这张图片"}
    ]}],
    "max_tokens": 500
}).encode()

req = urllib.request.Request(
    "http://localhost:30000/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req, timeout=120)
print(json.loads(resp.read())["choices"][0]["message"])
```

---

## 优化原理

本方案相比通用 vLLM 的 4 个核心优化:

1. **SM121 专用编译**: 从源码编译 vLLM + FlashInfer，针对 GB10 的 SM121 架构生成专用 CUDA kernel，注意力计算效率大幅提升
2. **MTP-2 投机解码**: 利用模型自带的 Multi-Token Prediction 权重，每步预测 3 个 token（接受 2.4-2.9 个），等效吞吐提升 40-50%，且无精度损失
3. **INT8 LM Head**: 运行时将 lm_head 层量化为 INT8，减少显存占用，为 KV cache 腾出空间
4. **Hybrid INT4+FP8 dispatch**: shared expert dense 层使用 FP8 替代 INT4，精度更高且在 GB10 上计算更快

---

## 参考

- 论坛讨论: https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-38-4-tok-s-patches-benchmark-included/365639
- 优化仓库: https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
- SM121 vLLM Docker: https://github.com/eugr/spark-vllm-docker
