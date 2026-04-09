# DGX Spark 双节点大模型推理部署

通过光缆 (ConnectX-7) 连接两台 DGX Spark (GB10)，使用 vLLM + Ray 实现跨节点 Tensor Parallelism 大模型推理。

## 快速开始

```bash
bash quick-start.sh <工作节点IP> <模型路径>
```

```bash
# 示例: 部署 Qwen3.5-122B (双节点 TP=2)
bash quick-start.sh 192.168.130.8 ~/models/Qwen3___5-122B-A10B-NVFP4

# 指定镜像
bash quick-start.sh 192.168.130.8 ~/models/YOUR_MODEL --image your-image:tag

# 仅校验不执行
bash quick-start.sh 192.168.130.8 ~/models/YOUR_MODEL --dry-run
```

脚本自动完成全部流程:

1. **预检校验** — Docker、镜像、模型文件、光口检测、高速网连通性
2. **同步镜像** — 通过高速网将 Docker 镜像传输到工作节点
3. **同步模型** — 通过高速网 rsync 模型到工作节点
4. **生成配置** — 自动检测量化方式、网络接口、NCCL/GLOO 参数
5. **启动服务** — Ray Head + Worker 组建集群，vLLM TP=2 推理
6. **验证部署** — 健康检查 + 推理测试

### 管理命令

```bash
bash quick-start.sh --status                    # 查看服务状态
bash quick-start.sh --stop 192.168.130.8        # 停止服务
docker logs -f --tail 50 vllm-spark-head        # 查看日志
```

### 选项

| 选项 | 说明 |
|------|------|
| `--image IMAGE` | 指定 Docker 镜像 (默认 ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603) |
| `--port PORT` | API 端口 (默认 30000) |
| `--max-len LEN` | 最大上下文长度 (默认 8192) |
| `--no-sync-model` | 跳过模型同步 |
| `--no-sync-image` | 跳过镜像同步 |
| `--dry-run` | 仅校验，不执行 |
| `--status` | 查看服务状态 |
| `--stop WORKER_IP` | 停止服务 |

## 自动检测

脚本自动检测以下配置，无需手动指定:

- **光口**: 遍历 ConnectX-7 四个口，找到有光缆连接 (carrier=1) 的接口
- **高速网 IP**: 从活跃光口读取已配置的 IP 地址
- **量化方式**: 从模型 config.json / hf_quant_config.json 自动识别
- **GLOO 接口**: 从管理网 IP 反推网口名
- **光口适配**: 两台光口名不同时自动切换 NCCL 检测模式

## 硬件

| 节点 | 角色 | GPU | 内存 | 互联 |
|------|------|-----|------|------|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 128 GiB 统一内存 | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 128 GiB 统一内存 | 200Gbps RoCE |

## 架构

```
spark01 (head)                    spark02 (worker)
+-----------------------+        +-----------------------+
|  Ray Head (6379)      |        |  Ray Worker           |
|  vLLM API (:30000)    |<------>|                       |
|  GB10 GPU             | RoCE   |  GB10 GPU             |
|  TP rank 0            | 200G   |  TP rank 1            |
+-----------------------+        +-----------------------+
```

## 已验证模型

| 模型 | 量化 | TP | 镜像 | 速度 |
|------|------|----|------|------|
| Qwen3.5-122B-A10B-NVFP4 | compressed-tensors | 2 | vllm-spark:v019-ngc2603 | ~17 t/s |
| Qwen3.5-35B-A3B-NVFP4 | modelopt_fp4 | 1 | sglang-dev-cu13-accel | ~30 t/s |

## 模型预设

`models/` 目录下提供了验证过的模型配置:

```bash
ls models/
# gemma4-26b-a4b.env         — Gemma 4 26B MoE (TP1)
# qwen3.5-122b-nvfp4.env     — Qwen3.5 122B NVFP4 (TP1)
# qwen3.5-122b-nvfp4-tp2.env — Qwen3.5 122B NVFP4 (TP2)
# qwen3.5-122b-fp8.env       — Qwen3.5 122B FP8 (TP2)
# qwen3.5-397b-int4.env      — Qwen3.5 397B INT4 (TP2)
# intel-122b-int4.env         — Intel INT4 AutoRound (TP1)
# redhatai-122b-nvfp4.env     — RedHatAI NVFP4 (TP1)
# wangzhang-122b-fp8.env      — abliterated FP8 (TP2)
# wangzhang-122b-nvfp4.env    — abliterated NVFP4 (TP1)
```

## API

兼容 OpenAI 格式:

```bash
curl http://192.168.130.16:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3___5-122B-A10B-NVFP4",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 500
  }'
```

## 目录结构

```
dgx-spark-multinode/
├── quick-start.sh              # 唯一入口
├── README.md
├── runtime/                    # 容器运行时配置
│   ├── docker-compose.yml      # vLLM head + worker 编排
│   ├── entrypoint.sh           # 容器入口 (TP1直连/TP2 Ray)
│   ├── .env.example            # 配置模板
│   └── patches/                # DGX Spark SM121 兼容补丁
├── models/                     # 模型预设 (.env 文件)
└── legacy/                     # 旧版 SGLang 部署脚本
```

## 前置准备

### SSH 免密

```bash
ssh-copy-id ai@192.168.130.16
ssh-copy-id ai@192.168.130.8
# Spark 之间也要互通
ssh ai@192.168.130.16 "ssh-copy-id ai@192.168.130.8"
```

### 高速网 IP

两台 DGX Spark 用光缆直连同一组 ConnectX-7 网口，手动配置 IP:

```bash
# spark01
sudo ip addr add 10.0.0.1/24 dev enp1s0f0np0
# spark02
sudo ip addr add 10.0.0.2/24 dev enp1s0f0np0
```

### Docker 镜像

```bash
# 从南大镜像拉取 (国内快)
docker pull ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603
# 或从 GitHub 原站
docker pull ghcr.io/bjk110/vllm-spark:v019-ngc2603
```

## 注意事项

- DGX Spark 是统一内存架构，122B 模型需双节点 TP=2
- 当前使用 NCCL TCP Socket (`NCCL_IB_DISABLE=1`)，后续可配置 RoCE
- 模型文件需在两台机器的相同路径下
- `runtime/docker-compose.yml` 配置了 `restart: unless-stopped`

## 致谢

- [bjk110/spark_vllm_docker](https://github.com/bjk110/spark_vllm_docker) — DGX Spark vLLM 适配和 SM121 补丁
- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理引擎
