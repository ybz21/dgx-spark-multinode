# vLLM Benchmark Report

- **Endpoint**: `http://192.168.130.8:30000`
- **Model**: `qwen3.5-122b-int4`
- **Timestamp**: 2026-04-13 15:10:01 +0800
- **Script**: `bench_llm.py`

## 1. Long-context needle-in-a-haystack

- Target prompt: **250,000** tokens (actual: 245,056)
- Needle: `MAGENTA-7834-GORILLA`

| Depth | Found | Prompt tok | TTFT (s) | Total (s) | Answer |
|------:|:-----:|-----------:|---------:|----------:|--------|
| 0% | ✅ | 245120 | 286.20 | 286.24 | MAGENTA-7834-GORILLA |
| 50% | ✅ | 245121 | 284.83 | 284.98 | MAGENTA-7834-GORILLA |
| 99% | ✅ | 245121 | 284.58 | 284.70 | MAGENTA-7834-GORILLA |

**Pass rate**: 3/3

## 2. Latency sweep (single request, greedy decode)

| Prompt tok | TTFT (s) | Total (s) | Out tok | Prefill tok/s | Decode tok/s |
|-----------:|---------:|----------:|--------:|--------------:|-------------:|
| 936 | 0.67 | 1.45 | 36 | 1388 | 46.2 |
| 7,879 | 3.97 | 4.77 | 35 | 1983 | 43.9 |
| 31,390 | 16.87 | 17.63 | 33 | 1861 | 43.3 |
| 122,629 | 97.92 | 98.74 | 33 | 1252 | 40.5 |
| 245,233 | 284.39 | 285.49 | 39 | 862 | 35.5 |

