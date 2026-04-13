# TTFT 瓶颈分析与上下文长度经验指南

> 基于 DGX Spark (GB10) 单机跑 Qwen3.5-122B A10B INT4-AutoRound + MTP-2 + FLASHINFER 的实测数据

## 1. 实测 TTFT 曲线（单请求、无并发、greedy）

| Prompt tok | TTFT | Prefill tok/s | 相对斜率 |
|----------:|-----:|--------------:|---------|
| 1,926 | **0.98s** | 1974 | — |
| 3,906 | **2.01s** | 1943 | 0.52 ms/tok |
| 5,890 | **2.83s** | 2082 | 0.41 ms/tok |
| 7,879 | **3.77s** | 2088 | 0.47 ms/tok |
| 11,806 | **5.72s** | 2065 | 0.50 ms/tok |
| 15,733 | **7.74s** | 2031 | 0.51 ms/tok |
| 23,587 | **12.15s** | 1941 | 0.56 ms/tok |
| 31,390 | **16.71s** | 1878 | 0.58 ms/tok |
| 47,098 | **27.05s** | 1741 | 0.66 ms/tok |
| 62,806 | **39.04s** | 1609 | 0.76 ms/tok |
| 94,171 | **66.98s** | 1406 | 0.89 ms/tok |
| 122,629 | **96.93s** | 1265 | 0.95 ms/tok |
| 245,233 | **284.39s** | 862 | 1.52 ms/tok |

**斜率（每新增 1k token 多花多少 TTFT）**在 8k 之前稳定在 ~0.5 ms/tok，之后单调上升，250k 时已是 1.5 ms/tok——典型的 Attention O(n²) 曲线。

## 2. 瓶颈归因

### 2.1 硬件物理上限
DGX Spark 的 GB10 是「workstation-class」集成 GPU：

| 项 | GB10 | H100 SXM | 比例 |
|---|---|---|---|
| Dense BF16 算力 | ~125 TFLOPS | ~990 TFLOPS | 0.13× |
| 内存带宽 | ~200 GB/s (LPDDR5X 统一内存) | 3350 GB/s (HBM3) | 0.06× |
| 显存类型 | 统一内存（CPU/GPU 共用） | 独立 HBM | — |

用 H100 比是「不公平」的比较，但能解释为何数据中心部署 TTFT 可以做到 <1s：他们的硬件不在一个量级。

### 2.2 Prefill 为什么这么慢
跑一次 prefill 需要对 prompt 里每个 token 都做完整的 122B forward pass（MoE 只激活 10B，但所有层都要过）。单请求 prefill 的理论下限主要受三个因素叠加：

1. **Weight memory read**：每层要从统一内存读激活的专家权重。INT4 压缩后全模型 67GB，layer-by-layer 按需读，~200 GB/s 带宽意味着纯权重读取的下限约 0.34s（全模型一次）。
2. **Attention O(n²)**：token 之间互相算 QK^T。短 prompt (≤8k) 时 attention 占比小，prefill 吞吐平稳在 ~2000 tok/s。长 prompt 攻击吞吐：64k 是 1609 tok/s，128k 是 1265 tok/s，250k 跌到 862 tok/s。
3. **Kernel launch + INT4 dequant overhead**：每层的 matmul 前要实时 dequant INT4 到 BF16，且 FLASHINFER/MoE 路由器内部有大量小 kernel。GB10 的 kernel launch overhead 相对算力更重。

### 2.3 MTP / FLASHINFER 不帮 TTFT
- **MTP 投机解码**：只加速 decode 阶段（省每 k 个 token 的 forward pass 次数）。Prefill 每个 prompt token 都要算一遍，投机用不上。
- **FLASHINFER**：长上下文 attention 后端，主要优化**内存访问模式**，降低常数项；它让 250k 还能跑（不 OOM），但不改变 O(n²) 量级。

### 2.4 当前 vLLM 配置可优化项
从启动日志看（grep 结果）：

```
enable_prefix_caching=False    ← 默认关，这是个大优化点
enable_chunked_prefill=True    ← 已开，帮并发不帮单请求 TTFT
Asynchronous scheduling enabled ← 已开
speculative_config=mtp num_spec_tokens=2  ← 已开，decode 加速
```

**Prefix Caching 是当前最大的漏网之鱼**。如果 prompt 有固定前缀（system prompt / RAG 检索出的相同文档 / 多轮对话的历史），启用后**相同前缀的 TTFT 从 ~50s 降到 <100ms**。启用方式：

```yaml
command:
  - serve
  - ...
  - --enable-prefix-caching  # 加这一行
```

代价：KV 按小 block 管理，多用 <5% 的 KV 内存，prompt 命中时省掉 prefill。

## 3. 经验指南 · 场景 → 可接受 TTFT → 最大 prompt

| 用户体验目标 | 可接受 TTFT | **建议最大 prompt** | 典型场景 |
|---|---|---|---|
| 即时响应 | < 1s | **2k tok** | 极短问答 |
| 聊天流畅 | < 3s | **6k tok** | 日常对话、短代码 |
| UI 有加载动画 | < 5s | **10k tok** | 单轮工具调用、简单 RAG |
| 可接受等待 | < 10s | **20k tok** | 单篇文章摘要、工具链 |
| 长文档场景 | < 20s | **40k tok** | 上限是「用户愿意多停一会儿」 |
| 异步/批处理 | < 60s | **90k tok** | 离线分析、报告生成 |
| 批处理 only | > 60s | 128k+ | 整本书、代码库 |

### 3.1 快速心算公式
在当前 GB10 单节点配置下：

```
TTFT(s) ≈ prompt_tokens × 1000 ÷ prefill_tok_per_s
       ≈ prompt_tokens ÷ 1800          (prompt ≤ 32k)
       ≈ prompt_tokens ÷ 1500          (prompt 32k-128k)
       ≈ prompt_tokens ÷ 1000          (prompt > 128k)
```

倒过来，想要 TTFT < X 秒：

```
允许的 prompt tokens ≈ X × 1800    (当 X ≤ 18s)
```

## 4. 如何把 TTFT 降下来（按 ROI 排序）

### 推荐（立刻可做）
1. **启用 Prefix Caching** — 加 `--enable-prefix-caching`，命中场景 TTFT 近零
2. **应用层切块** — 不要一次塞 100k，先用小模型/embedding 做 RAG，只把最相关的 8k-16k 喂给 122B
3. **减少 system prompt 长度** — system prompt 每轮都会占用 prefill（prefix caching 开后可免）
4. **降低 max_model_len** — 单纯用 32k 配置，日常用没差别，但 KV 调度块变小，极少数场景 TTFT 略改善

### 进阶（改架构）
5. **跨节点 Tensor Parallelism**（本 repo `../quick-start.sh`） — 双卡并行，prefill 近似砍半。代价：需要两台 DGX Spark + ConnectX 光纤
6. **降配模型** — 换 Qwen3.5-35B A3B 或更小。prefill 快 3-5 倍。
7. **换硬件** — H100/H200 数据中心部署，prefill 吞吐 ~20k tok/s，128k TTFT ~6s。

### 不推荐
- **去掉 MTP**：不影响 prefill，反而会让 decode 慢下来，净负收益。
- **降 `--gpu-memory-utilization`**：对 TTFT 影响小，只缩 KV cache 并发上限。
- **换 attention backend**：FLASHINFER 就是长上下文最好的选择。

## 5. 一句话结论

**DGX Spark 单节点 + 122B A10B 的 prefill 吞吐上限 ~2000 tok/s（短 prompt），随上下文增长单调下降到 ~1000 tok/s（128k+）**。想要 <5s TTFT 的交互体验，**prompt 控制在 10k 以内**；想跑 RAG 或长上下文，**要么启用 prefix caching，要么接受异步 UX**。

## 附录 A. 测试方法

```bash
# 细粒度 sweep
python3 bench_llm.py --base http://<host>:30000 \
    --skip-needle --skip-stress \
    --latency-sizes 2000,4000,6000,8000,12000,16000,24000,32000,48000,64000,96000 \
    --out bench_ttft_sweep.md
```

原始结果：[`bench_ttft_sweep.md`](./bench_ttft_sweep.md)、[`bench_ttft_sweep.json`](./bench_ttft_sweep.json)
