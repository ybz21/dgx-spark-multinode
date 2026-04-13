#!/usr/bin/env python3
"""bench_llm.py - vLLM OpenAI-compatible endpoint benchmark.

Tests:
  1. Long-context needle-in-haystack (~128k prompt, multiple depths)
  2. Latency sweep at varying prompt sizes (TTFT, prefill/decode tok/s)
  3. Concurrency stress (short prompts, multiple levels)

Outputs a markdown report + raw JSON.

Usage:
  python3 bench_llm.py \
      --base http://192.168.130.12:30000 \
      --model qwen3.5-122b-int4 \
      --out bench_report.md
"""
import argparse, json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

NEEDLE_SECRET = "MAGENTA-7834-GORILLA"
FILLER_CHUNK = (
    "In a distant research lab, scientists studied the behavior of migrating "
    "birds across five continents. They tracked seasonal patterns, tagged "
    "individuals, and correlated flight paths with magnetic field data. "
    "Their instruments recorded wind speeds, humidity, and geomagnetic "
    "fluctuations across thousands of nights. "
)


def tokenize(base, text):
    r = requests.post(f"{base}/tokenize", json={"prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["count"]


def stream_chat(base, model, messages, max_tokens=128, temperature=0.0,
                enable_thinking=False):
    """Stream chat, return dict with ttft, total, out_tokens, prompt_tokens,
    content, reasoning."""
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "stream": True, "temperature": temperature,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    t0 = time.time()
    ttft = None
    out_tokens = 0
    prompt_tokens = 0
    content_parts = []
    reasoning_parts = []
    err = None
    try:
        with requests.post(f"{base}/v1/chat/completions", json=body,
                           stream=True, timeout=3600) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                u = obj.get("usage")
                if u:
                    out_tokens = u.get("completion_tokens", out_tokens)
                    prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                for ch in obj.get("choices", []):
                    delta = ch.get("delta") or {}
                    c = delta.get("content") or ""
                    rc = (delta.get("reasoning_content") or
                          delta.get("reasoning") or "")
                    if (c or rc) and ttft is None:
                        ttft = time.time() - t0
                    if c:
                        content_parts.append(c)
                    if rc:
                        reasoning_parts.append(rc)
    except Exception as e:
        err = str(e)
    total = time.time() - t0
    return {
        "ttft": ttft, "total": total, "out_tokens": out_tokens,
        "prompt_tokens": prompt_tokens,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "text": "".join(content_parts),  # backward-compat
        "error": err,
    }


def build_filler(base, target_tokens):
    per = tokenize(base, FILLER_CHUNK)
    n = target_tokens // per + 2
    text = FILLER_CHUNK * n
    got = tokenize(base, text)
    # binary-ish trim
    while got > target_tokens and len(text) > 100:
        ratio = target_tokens / got
        text = text[: int(len(text) * ratio * 0.99)]
        got = tokenize(base, text)
    return text, got


def needle_test(base, model, target_tokens, depths):
    print(f"  Building {target_tokens}-token haystack...")
    haystack, actual = build_filler(base, target_tokens - 200)
    print(f"  Haystack built: {actual} tokens")
    results = []
    for d in depths:
        idx = int(len(haystack) * d)
        if 0 < idx < len(haystack):
            nxt = haystack.find(". ", idx)
            idx = nxt + 2 if nxt != -1 else idx
        doc = haystack[:idx] + f" {NEEDLE_SECRET} is the secret passcode. " + haystack[idx:]
        msg = [{"role": "user", "content":
                "Read the document below and find the secret passcode. "
                "Answer with just the passcode (letters, dash, digits).\n\n"
                f"[DOCUMENT]\n{doc}\n[/DOCUMENT]\n\nPasscode:"}]
        t_start = time.time()
        res = stream_chat(base, model, msg, max_tokens=48)
        elapsed = time.time() - t_start
        ans = res["text"].strip()
        found = all(s in ans.upper() for s in ["MAGENTA", "7834", "GORILLA"])
        print(f"  depth={d*100:5.1f}%  found={found}  ttft={res['ttft']}  total={elapsed:.1f}s  ans={ans[:80]!r}")
        results.append({
            "depth": d, "found": found, "answer": ans[:200],
            "ttft": res["ttft"], "total": res["total"],
            "prompt_tokens": res["prompt_tokens"], "out_tokens": res["out_tokens"],
            "error": res["error"],
        })
    return {"target_tokens": target_tokens, "actual_tokens": actual, "results": results}


def latency_sweep(base, model, sizes):
    rows = []
    for n in sizes:
        print(f"  prompt~{n} tok...", flush=True)
        doc, actual = build_filler(base, n - 80)
        msg = [{"role": "user", "content": doc + "\n\nSummarize the key theme in <=40 words."}]
        r = stream_chat(base, model, msg, max_tokens=160)
        decode_time = (r["total"] - (r["ttft"] or 0))
        out_tps = (r["out_tokens"] / decode_time) if decode_time > 0 and r["out_tokens"] else 0.0
        prefill_tps = (r["prompt_tokens"] / r["ttft"]) if r["ttft"] else 0.0
        print(f"    prompt_tok={r['prompt_tokens']}  ttft={r['ttft']}  total={r['total']:.1f}s  "
              f"out_tok={r['out_tokens']}  prefill={prefill_tps:.0f}tok/s  decode={out_tps:.1f}tok/s")
        rows.append({
            "target": n, "prompt_tokens": r["prompt_tokens"],
            "ttft": r["ttft"], "total": r["total"],
            "out_tokens": r["out_tokens"],
            "prefill_tps": prefill_tps, "decode_tps": out_tps,
            "error": r["error"],
        })
    return rows


def stress_test(base, model, levels, prompt_tokens, max_out, reqs_per_level):
    print(f"  Building {prompt_tokens}-token prompt...")
    doc, actual = build_filler(base, prompt_tokens)
    print(f"  Prompt: {actual} tokens")

    def one(i):
        msg = [{"role":"user","content": doc + f"\n\nReply with 'ack-{i}' then write a short poem about birds."}]
        return stream_chat(base, model, msg, max_tokens=max_out, temperature=0.7)

    rows = []
    for c in levels:
        n = reqs_per_level
        print(f"  concurrency={c}  requests={n}...", flush=True)
        t0 = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(one, i) for i in range(n)]
            for f in as_completed(futs):
                results.append(f.result())
        dur = time.time() - t0
        succ = [r for r in results if r["error"] is None and r["ttft"] is not None]
        fail = len(results) - len(succ)
        ttfts = [r["ttft"] for r in succ]
        totals = [r["total"] for r in succ]
        tot_out = sum(r["out_tokens"] for r in succ)
        def q(vals, pct):
            if not vals: return None
            s = sorted(vals)
            k = max(0, min(len(s)-1, int(round((pct/100)*(len(s)-1)))))
            return s[k]
        row = {
            "concurrency": c, "requests": n, "duration": dur,
            "success": len(succ), "fail": fail,
            "p50_ttft": q(ttfts,50), "p95_ttft": q(ttfts,95),
            "p50_total": q(totals,50), "p95_total": q(totals,95),
            "total_out_tokens": tot_out,
            "aggregate_out_tps": tot_out / dur if dur > 0 else 0,
            "rps": n / dur if dur > 0 else 0,
        }
        print(f"    dur={dur:.1f}s  ok={len(succ)}/{n}  p50_ttft={row['p50_ttft']}  "
              f"agg_out={row['aggregate_out_tps']:.1f}tok/s  rps={row['rps']:.2f}")
        rows.append(row)
    return {"prompt_tokens": actual, "max_output_tokens": max_out, "levels": rows}


def fmt(v, f="{:.2f}", dash="-"):
    return dash if v is None else f.format(v)


def render_report(meta, needle, latency, stress):
    lines = [
        "# vLLM 基准测试报告",
        "",
        f"- **服务端点**：`{meta['base']}`",
        f"- **模型名**：`{meta['model']}`",
        f"- **时间戳**：{meta['ts']}",
        f"- **脚本**：`{meta['script']}`",
        "",
    ]
    if needle:
        lines += [
            "## 1. 长上下文大海捞针（needle-in-a-haystack）",
            "",
            f"- 目标 prompt 长度：**{needle['target_tokens']:,}** tokens "
            f"（实际：{needle['actual_tokens']:,}）",
            f"- 针：`{NEEDLE_SECRET}`",
            "",
            "| 深度 | 找到 | Prompt tok | TTFT（秒） | 总耗时（秒） | 回答 |",
            "|----:|:---:|-----------:|---------:|-----------:|--------|",
        ]
        for r in needle["results"]:
            ans = (r["answer"] or "").replace("|", "/").replace("\n", " ")[:80]
            lines.append(
                f"| {r['depth']*100:.0f}% | {'✅' if r['found'] else '❌'} | "
                f"{r.get('prompt_tokens') or '-'} | {fmt(r['ttft'])} | "
                f"{fmt(r['total'])} | {ans} |"
            )
        passed = sum(1 for r in needle["results"] if r["found"])
        lines += ["", f"**通过率**：{passed}/{len(needle['results'])}", ""]
    if latency:
        lines += [
            "## 2. 延迟扫描（单请求、贪心解码）",
            "",
            "| Prompt tok | TTFT（秒） | 总耗时（秒） | 输出 tok | Prefill tok/s | Decode tok/s |",
            "|-----------:|---------:|-----------:|--------:|-------------:|------------:|",
        ]
        for r in latency:
            lines.append(
                f"| {r['prompt_tokens']:,} | {fmt(r['ttft'])} | "
                f"{fmt(r['total'])} | {r['out_tokens']} | "
                f"{fmt(r['prefill_tps'], '{:.0f}')} | {fmt(r['decode_tps'], '{:.1f}')} |"
            )
        lines.append("")
    if stress:
        lines += [
            f"## 3. 并发压测"
            f"（prompt={stress['prompt_tokens']} tok，max_output={stress['max_output_tokens']} tok）",
            "",
            "| 并发 | 请求数 | 耗时（秒） | 成功 | 失败 | p50 TTFT（秒） | p95 TTFT（秒） | "
            "p50 总耗时（秒） | 聚合输出 tok/s | RPS |",
            "|----:|----:|---------:|---:|----:|-------------:|-------------:|"
            "----------------:|--------------:|----:|",
        ]
        for r in stress["levels"]:
            lines.append(
                f"| {r['concurrency']} | {r['requests']} | {r['duration']:.1f} | "
                f"{r['success']} | {r['fail']} | {fmt(r['p50_ttft'])} | "
                f"{fmt(r['p95_ttft'])} | {fmt(r['p50_total'])} | "
                f"{r['aggregate_out_tps']:.1f} | {r['rps']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://192.168.130.12:30000",
                   help="Endpoint root (no /v1 suffix)")
    p.add_argument("--model", default="qwen3.5-122b-int4")
    p.add_argument("--out", default="bench_report.md")
    p.add_argument("--json", default="bench_report.json")
    p.add_argument("--needle-tokens", type=int, default=125000)
    p.add_argument("--needle-depths", default="0,0.5,0.99",
                   help="Comma-separated fractional depths")
    p.add_argument("--latency-sizes", default="1000,8000,32000,125000",
                   help="Comma-separated prompt token targets")
    p.add_argument("--stress-levels", default="1,4,8")
    p.add_argument("--stress-prompt-tokens", type=int, default=512)
    p.add_argument("--stress-max-out", type=int, default=128)
    p.add_argument("--stress-reqs", type=int, default=16)
    p.add_argument("--skip-needle", action="store_true")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--skip-stress", action="store_true")
    args = p.parse_args()

    base = args.base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    print(f"Endpoint: {base}  Model: {args.model}")
    t = tokenize(base, "probe")
    print(f"Tokenizer probe ok ({t} tokens for 'probe')")

    meta = {"base": base, "model": args.model,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "script": os.path.basename(__file__)}

    needle = latency = stress = None

    if not args.skip_needle:
        print("\n[1/3] Needle-in-a-haystack")
        depths = [float(x) for x in args.needle_depths.split(",") if x.strip()]
        needle = needle_test(base, args.model, args.needle_tokens, depths)

    if not args.skip_latency:
        print("\n[2/3] Latency sweep")
        sizes = [int(x) for x in args.latency_sizes.split(",") if x.strip()]
        latency = latency_sweep(base, args.model, sizes)

    if not args.skip_stress:
        print("\n[3/3] Concurrency stress")
        levels = [int(x) for x in args.stress_levels.split(",") if x.strip()]
        stress = stress_test(base, args.model, levels,
                             args.stress_prompt_tokens, args.stress_max_out,
                             args.stress_reqs)

    md = render_report(meta, needle, latency, stress)
    with open(args.out, "w") as f:
        f.write(md)
    with open(args.json, "w") as f:
        json.dump({"meta": meta, "needle": needle, "latency": latency, "stress": stress},
                  f, indent=2, ensure_ascii=False)
    print(f"\nReport: {args.out}\nRaw JSON: {args.json}")


if __name__ == "__main__":
    main()
