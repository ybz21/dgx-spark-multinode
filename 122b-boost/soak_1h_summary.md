# Soak test summary

- **Start**: 2026-04-13 14:43:09
- **End**:   2026-04-13 15:43:16


- **Endpoint**: `http://192.168.130.12:30000`
- **Model**: `qwen3.5-122b-int4`
- **Concurrency**: 4
- **Elapsed**: 1.00 h / target 1.00 h
- **Throughput**: 0.39 rps · success 1393/1393 (100.00%)
- **Rolling p50 TTFT**: 2.12s  ·  p95 19.75s  ·  decode 9.2 tok/s
- **Correctness probes**: pass 36  ·  fail 0

## Per-kind latency

| Kind | N | p50 TTFT | p95 TTFT |
|------|--:|---------:|---------:|
| short | 964 | 0.98s | 17.54s |
| medium | 276 | 2.64s | 19.62s |
| long | 117 | 19.17s | 31.27s |
| probe | 36 | 0.90s | 17.69s |

## Hourly drift

| Hour | N | Fails | p50 TTFT | p95 TTFT | Decode tok/s |
|-----:|--:|------:|---------:|---------:|-------------:|
| 0 | 1389 | 0 | 2.07s | 19.75s | 9.2 |
| 1 | 4 | 0 | 19.33s | 20.55s | 14.2 |

