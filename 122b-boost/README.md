# Qwen3.5-122B-A10B AutoRound INT4 · 单节点 Boost

单节点 DGX Spark (GB10) 跑 Qwen3.5-122B-A10B INT4 量化版，vLLM + FLASHINFER + **MTP 投机解码**。

和本仓库根目录的 **多节点 TP=2 NVFP4** 方案互为补充：

|  | 多节点 (../) | **本方案 (122b-boost/)** |
|---|---|---|
| 模型量化 | NVFP4 | INT4 AutoRound (Intel) |
| 部署 | 2× DGX Spark，TP=2 over ConnectX-7 | **单节点** |
| 跨机通信 | Ray + NCCL | 无 |
| MTP 投机 | 否 | **是 (k=2)** |
| 上下文 | 32K | **128K** |
| 首 token 延迟 @ 32k | — | 17s |
| Decode 吞吐 | — | 38–46 tok/s |
| 适合场景 | 跨节点显存扩展 | 单机极致吞吐 + 长上下文 |

参考上游：<https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4>

## 目录内容

| 文件 | 作用 |
|---|---|
| [`README.md`](./README.md) | 本文件，方案总览 |
| [`部署指南.md`](./部署指南.md) | 从零部署、参数调优、故障排查 |
| [`docker-compose.yaml`](./docker-compose.yaml) | 固化的服务定义（单容器 host 网络） |
| [`bench_llm.py`](./bench_llm.py) | 基准测试脚本（needle + latency + 并发压测） |
| [`bench_report.md`](./bench_report.md) | 最近一次基准结果（Markdown） |
| [`bench_report.json`](./bench_report.json) | 同上的原始 JSON |
| [`soak_test.py`](./soak_test.py) | **长久稳定性测试**（混合负载 + 正确性探针 + 分时漂移） |

## 快速使用

**前置**：模型 `Qwen3.5-122B-A10B-int4-AutoRound-Intel` 放在 `~/models/` 下（72GB），镜像 `vllm-qwen35-v2` 已就绪。

```bash
# 1. 部署（目标机器上）
scp docker-compose.yaml ai@<host>:~/lm_scripts/
ssh ai@<host> 'cd ~/lm_scripts && docker compose up -d'

# 2. 等待健康（≈ 10–15 min，加载 72GB 权重 + 编译 FLASHINFER）
ssh ai@<host> 'docker ps --filter name=vllm-qwen35 --format "{{.Status}}"'

# 3. 调用
curl http://<host>:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-122b-int4","messages":[{"role":"user","content":"你好"}]}'

# 4. 基准
python3 bench_llm.py --base http://<host>:30000 --out bench_report.md
```

详见 [部署指南](./部署指南.md)。

## 当前已部署节点

| 节点 | IP | 状态 |
|---|---|---|
| spark-c915 | `192.168.130.12` | 运行中 · `http://192.168.130.12:30000/v1` |

## 关键参数速查

- 端口：`30000`（host network）
- 模型名：`qwen3.5-122b-int4`
- 上下文上限：**131072** (128K)
- 显存预算：`--gpu-memory-utilization 0.90`（GB10 统一内存 128GB，约 101GB 被占）
- 投机解码：`mtp` · `num_speculative_tokens=2`（模型只有 1 层 MTP，再调大收益递减）
- Tool calling：`qwen3_coder` parser
- Reasoning：`qwen3` parser，关闭 thinking 传 `chat_template_kwargs.enable_thinking=false`
