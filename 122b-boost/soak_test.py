#!/usr/bin/env python3
"""soak_test.py - long-running stability / soak test for vLLM endpoint.

Runs a mixed workload (short / medium / long prompts + correctness probes) at
constant concurrency over a long duration. Captures rolling p50/p95, per-hour
drift, error categories, and correctness regressions.

Outputs in real time:
  <out_dir>/status.json   — machine-readable snapshot, overwritten each cycle
  <out_dir>/status.md     — human-friendly snapshot, overwritten each cycle
  <out_dir>/events.log    — append-only log of errors + probe failures
  <out_dir>/summary.md    — final report, written on exit

Ctrl-C (SIGINT) or SIGTERM → graceful shutdown + summary.

Usage:
  python3 soak_test.py --duration 4h --concurrency 4
  python3 soak_test.py --duration 24h --out-dir soak_20260413
"""
import argparse
import json
import os
import random
import re
import signal
import statistics
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_llm import stream_chat, build_filler, tokenize  # noqa: E402

# ---------- workload definitions ----------

# Each kind: (prompt_target_tokens, max_out_tokens, weight)
KINDS = {
    "short":  {"prompt": 512,   "max_out": 128, "weight": 70, "think": False},
    "medium": {"prompt": 4096,  "max_out": 256, "weight": 20, "think": False},
    "long":   {"prompt": 32000, "max_out": 200, "weight":  8, "think": False},
    "probe":  {"prompt": 0,     "max_out": 40,  "weight":  2, "think": False},
}

# (question, substring that must appear in answer; case-insensitive)
PROBES = [
    ("Compute 47 + 83. Reply with only the number.", "130"),
    ("What is 12 * 9? Reply with only the number.", "108"),
    ("Spell 'banana' backwards. Reply with only that word.", "ananab"),
    ("What is the capital of France? One word.", "paris"),
    ("Reply with exactly this token: ECHO-42-OK.", "echo-42-ok"),
]


# ---------- shared state ----------

class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.start = time.time()
        self.total = 0
        self.success = 0
        self.failed = 0
        self.by_kind = Counter()          # kind -> count
        self.ok_by_kind = Counter()
        self.errors = Counter()           # error signature -> count
        self.recent_errors = deque(maxlen=40)   # (ts, kind, signature)
        self.ttfts = deque(maxlen=2000)   # rolling (kind, ttft)
        self.decodes = deque(maxlen=2000) # rolling (kind, decode_tps)
        self.probe_pass = 0
        self.probe_fail = 0
        self.probe_failures = deque(maxlen=50)   # (ts, question, answer)
        self.hourly = {}                  # hour_idx -> {n, fails, ttfts[], decodes[]}

    def hour_key(self, t):
        return int((t - self.start) // 3600)

    def add(self, kind, res):
        with self.lock:
            self.total += 1
            self.by_kind[kind] += 1
            hk = self.hour_key(time.time())
            h = self.hourly.setdefault(hk, {"n": 0, "fails": 0,
                                             "ttfts": [], "decodes": []})
            h["n"] += 1
            if res.get("error") or res.get("ttft") is None:
                self.failed += 1
                h["fails"] += 1
                sig = self._sig(res.get("error") or "ttft=None")
                self.errors[sig] += 1
                self.recent_errors.append((time.time(), kind, sig))
            else:
                self.success += 1
                self.ok_by_kind[kind] += 1
                self.ttfts.append((kind, res["ttft"]))
                h["ttfts"].append(res["ttft"])
                dec_time = res["total"] - res["ttft"]
                if dec_time > 0 and res.get("out_tokens"):
                    tps = res["out_tokens"] / dec_time
                    self.decodes.append((kind, tps))
                    h["decodes"].append(tps)

    def add_probe(self, q, expect, ans):
        ok = expect.lower() in (ans or "").lower()
        with self.lock:
            if ok:
                self.probe_pass += 1
            else:
                self.probe_fail += 1
                self.probe_failures.append((time.time(), q, (ans or "")[:200]))
        return ok

    @staticmethod
    def _sig(err):
        # collapse noisy numbers/addresses so we get a small set of categories
        s = str(err)[:200]
        s = re.sub(r"0x[0-9a-fA-F]+", "0x?", s)
        s = re.sub(r"\b\d{3,}\b", "N", s)
        return s

    def snapshot(self):
        with self.lock:
            now = time.time()
            dur = now - self.start
            def pct(vals, p):
                if not vals:
                    return None
                s = sorted(vals)
                k = max(0, min(len(s) - 1,
                               int(round((p / 100) * (len(s) - 1)))))
                return s[k]
            rolling_ttfts = [t for _, t in self.ttfts]
            rolling_decs = [t for _, t in self.decodes]
            per_kind_ttft = {}
            for k in KINDS:
                vals = [t for kk, t in self.ttfts if kk == k]
                per_kind_ttft[k] = {
                    "n": len(vals),
                    "p50": pct(vals, 50), "p95": pct(vals, 95),
                }
            hourly = {}
            for hk, h in sorted(self.hourly.items()):
                hourly[hk] = {
                    "n": h["n"], "fails": h["fails"],
                    "p50_ttft": pct(h["ttfts"], 50),
                    "p95_ttft": pct(h["ttfts"], 95),
                    "mean_decode_tps": (statistics.mean(h["decodes"])
                                        if h["decodes"] else None),
                }
            return {
                "elapsed_s": dur,
                "elapsed_h": dur / 3600,
                "total": self.total, "success": self.success,
                "failed": self.failed,
                "rps": self.total / dur if dur > 0 else 0,
                "success_rate": (self.success / self.total) if self.total else 1,
                "by_kind": dict(self.by_kind),
                "ok_by_kind": dict(self.ok_by_kind),
                "rolling_p50_ttft": pct(rolling_ttfts, 50),
                "rolling_p95_ttft": pct(rolling_ttfts, 95),
                "rolling_mean_decode_tps": (statistics.mean(rolling_decs)
                                            if rolling_decs else None),
                "per_kind_ttft": per_kind_ttft,
                "hourly": hourly,
                "probe_pass": self.probe_pass, "probe_fail": self.probe_fail,
                "top_errors": dict(self.errors.most_common(10)),
                "recent_probe_failures": [
                    {"ts": t, "q": q, "ans": a}
                    for t, q, a in list(self.probe_failures)[-10:]
                ],
                "recent_errors": [
                    {"ts": t, "kind": k, "sig": s}
                    for t, k, s in list(self.recent_errors)[-10:]
                ],
            }


# ---------- prompt corpus (cached) ----------

class Corpus:
    """Pre-build filler texts for each kind so workers don't hit /tokenize
    repeatedly during the run."""

    def __init__(self, base):
        self.base = base
        self.prompts = {}  # kind -> doc_text
        self.actual_tokens = {}

    def prepare(self):
        for kind, cfg in KINDS.items():
            if kind == "probe":
                continue
            target = cfg["prompt"]
            print(f"  preparing {kind} prompt (~{target} tok)...",
                  flush=True)
            doc, actual = build_filler(self.base, max(200, target - 80))
            self.prompts[kind] = doc
            self.actual_tokens[kind] = actual


# ---------- workload ----------

def pick_kind():
    kinds = list(KINDS.keys())
    weights = [KINDS[k]["weight"] for k in kinds]
    return random.choices(kinds, weights=weights, k=1)[0]


def make_request(kind, corpus, iter_id):
    if kind == "probe":
        q, expect = random.choice(PROBES)
        msg = [{"role": "user", "content": q}]
        return msg, {"probe_expect": expect, "probe_q": q}
    doc = corpus.prompts[kind]
    instr = ("\n\nIn one short sentence, name the main topic discussed above."
             if kind != "short"
             else f"\n\nReply starting with 'ack-{iter_id}' then a single line."
             )
    msg = [{"role": "user", "content": doc + instr}]
    return msg, {}


def worker_loop(wid, args, stats, corpus, stop_event):
    rng = random.Random(wid * 1009 + int(time.time()))
    i = 0
    while not stop_event.is_set():
        i += 1
        kind = pick_kind()
        cfg = KINDS[kind]
        messages, meta = make_request(kind, corpus, f"{wid}-{i}")
        t0 = time.time()
        try:
            res = stream_chat(args.base, args.model, messages,
                              max_tokens=cfg["max_out"],
                              temperature=0.7 if kind == "short" else 0.3,
                              enable_thinking=cfg["think"])
        except Exception as e:
            res = {"error": f"exception: {e}", "ttft": None, "total": 0,
                   "out_tokens": 0, "prompt_tokens": 0, "content": "",
                   "reasoning": "", "text": ""}
        stats.add(kind, res)
        if kind == "probe":
            stats.add_probe(meta["probe_q"], meta["probe_expect"],
                            res.get("content"))
        if stop_event.is_set():
            break
        # small jitter so workers don't sync up
        time.sleep(rng.uniform(0.05, 0.4))


# ---------- reporting ----------

def fmt_ttft(v): return "-" if v is None else f"{v:.2f}s"
def fmt_tps(v):  return "-" if v is None else f"{v:.1f}"


def render_status_md(snap, args):
    lines = [
        "# 稳定性测试状态",
        "",
        f"- **服务端点**：`{args.base}`",
        f"- **模型名**：`{args.model}`",
        f"- **并发数**：{args.concurrency}",
        f"- **已运行**：{snap['elapsed_h']:.2f} 小时 "
        f"/ 目标 {args.duration / 3600:.2f} 小时",
        f"- **吞吐**：{snap['rps']:.2f} rps · "
        f"成功 {snap['success']}/{snap['total']} "
        f"（{snap['success_rate']*100:.2f}%）",
        f"- **滚动 p50 TTFT**：{fmt_ttft(snap['rolling_p50_ttft'])}  ·  "
        f"p95 {fmt_ttft(snap['rolling_p95_ttft'])}  ·  "
        f"解码 {fmt_tps(snap['rolling_mean_decode_tps'])} tok/s",
        f"- **正确性探针**：通过 {snap['probe_pass']}  ·  "
        f"失败 {snap['probe_fail']}",
        "",
        "## 分负载延迟",
        "",
        "| 负载类型 | 请求数 | p50 TTFT | p95 TTFT |",
        "|------|--:|---------:|---------:|",
    ]
    for k, v in snap["per_kind_ttft"].items():
        lines.append(f"| {k} | {v['n']} | {fmt_ttft(v['p50'])} | "
                     f"{fmt_ttft(v['p95'])} |")
    lines += ["", "## 分时漂移",
              "",
              "| Hour | 请求数 | 失败 | p50 TTFT | p95 TTFT | Decode tok/s |",
              "|-----:|--:|------:|---------:|---------:|-------------:|"]
    for hk, h in snap["hourly"].items():
        lines.append(
            f"| {hk} | {h['n']} | {h['fails']} | {fmt_ttft(h['p50_ttft'])} | "
            f"{fmt_ttft(h['p95_ttft'])} | {fmt_tps(h['mean_decode_tps'])} |"
        )
    if snap["top_errors"]:
        lines += ["", "## 主要错误签名", ""]
        for sig, n in snap["top_errors"].items():
            lines.append(f"- **{n}×** `{sig[:140]}`")
    if snap["recent_probe_failures"]:
        lines += ["", "## 最近探针失败", ""]
        for p in snap["recent_probe_failures"]:
            lines.append(
                f"- `{time.strftime('%H:%M:%S', time.localtime(p['ts']))}` "
                f"**问题**：{p['q']}  **回答**：{p['ans'][:120]!r}")
    return "\n".join(lines) + "\n"


def snapshot_loop(args, stats, stop_event):
    os.makedirs(args.out_dir, exist_ok=True)
    status_json = os.path.join(args.out_dir, "status.json")
    status_md = os.path.join(args.out_dir, "status.md")
    events_log = os.path.join(args.out_dir, "events.log")
    last_err_seen = 0
    last_probe_seen = 0
    while not stop_event.is_set():
        time.sleep(args.status_interval)
        snap = stats.snapshot()
        with open(status_json, "w") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False)
        with open(status_md, "w") as f:
            f.write(render_status_md(snap, args))
        # append new events
        with stats.lock:
            new_errs = list(stats.recent_errors)[last_err_seen:]
            last_err_seen = len(stats.recent_errors)
            new_probes = list(stats.probe_failures)[last_probe_seen:]
            last_probe_seen = len(stats.probe_failures)
        with open(events_log, "a") as f:
            for t, k, s in new_errs:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))}"
                        f" ERROR kind={k} sig={s[:200]!r}\n")
            for t, q, a in new_probes:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))}"
                        f" PROBE_FAIL q={q!r} a={a[:120]!r}\n")
        # one-line console update
        print(f"[{snap['elapsed_h']:5.2f}h] "
              f"req={snap['total']} ok={snap['success']} "
              f"fail={snap['failed']} probe={snap['probe_pass']}/"
              f"{snap['probe_pass']+snap['probe_fail']} "
              f"p50={fmt_ttft(snap['rolling_p50_ttft'])} "
              f"p95={fmt_ttft(snap['rolling_p95_ttft'])} "
              f"dec={fmt_tps(snap['rolling_mean_decode_tps'])}tps",
              flush=True)


def render_summary_md(snap, args):
    lines = render_status_md(snap, args).split("\n")
    lines[0] = "# 稳定性测试总结报告"
    hdr = [
        "",
        f"- **开始时间**：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - snap['elapsed_s']))}",
        f"- **结束时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    return "\n".join(lines[:1] + hdr + lines[1:]) + "\n"


# ---------- main ----------

def parse_duration(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])?", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad duration: {s}")
    n = float(m.group(1))
    u = m.group(2) or "s"
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://192.168.130.12:30000")
    p.add_argument("--model", default="qwen3.5-122b-int4")
    p.add_argument("--duration", type=parse_duration, default="4h",
                   help="e.g. 30m, 4h, 24h")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--status-interval", type=int, default=60)
    p.add_argument("--out-dir", default="soak_out")
    p.add_argument("--disable-long", action="store_true",
                   help="Skip 32k-prompt kind (saves time if prefill slow)")
    args = p.parse_args()

    if args.base.rstrip("/").endswith("/v1"):
        args.base = args.base.rstrip("/")[:-3]
    args.base = args.base.rstrip("/")

    if args.disable_long:
        KINDS["long"]["weight"] = 0

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Endpoint: {args.base}   Model: {args.model}")
    print(f"Duration: {args.duration:.0f}s ({args.duration / 3600:.2f}h)   "
          f"Concurrency: {args.concurrency}")
    print(f"Out dir: {os.path.abspath(args.out_dir)}")

    # sanity
    t = tokenize(args.base, "probe")
    print(f"Tokenizer probe ok ({t} tokens)")

    corpus = Corpus(args.base)
    print("Preparing prompt corpus...")
    corpus.prepare()

    stats = Stats()
    stop_event = threading.Event()

    def shutdown(signum, frame):
        print(f"\nSignal {signum} — draining workers...")
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # spin up workers
    reporter = threading.Thread(
        target=snapshot_loop, args=(args, stats, stop_event), daemon=True)
    reporter.start()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker_loop, i, args, stats, corpus, stop_event)
                   for i in range(args.concurrency)]
        deadline = time.time() + args.duration
        try:
            while time.time() < deadline and not stop_event.is_set():
                time.sleep(1)
        finally:
            stop_event.set()
            for f in futures:
                try:
                    f.result(timeout=60)
                except Exception as e:
                    print(f"worker exception: {e}")

    # final snapshot + summary
    snap = stats.snapshot()
    with open(os.path.join(args.out_dir, "status.json"), "w") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(render_summary_md(snap, args))
    print("\n=== SUMMARY ===")
    print(f"Elapsed: {snap['elapsed_h']:.2f}h   "
          f"Total: {snap['total']}   "
          f"Success rate: {snap['success_rate']*100:.2f}%   "
          f"Probes: {snap['probe_pass']}/"
          f"{snap['probe_pass']+snap['probe_fail']}")
    print(f"Report: {os.path.join(args.out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
