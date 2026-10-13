#!/usr/bin/env python3
"""Throughput benchmark for the local Qwen3.8-27B-w8a8 vLLM server.

Measures:
  * single-stream decode throughput (tokens/s) -- the headline number
  * time-to-first-token (TTFT)
  * aggregate output throughput across a concurrency sweep

Token counts come from the server's own `usage.completion_tokens`
(which includes reasoning tokens), so they are exact.

Usage:
    python3 benchmark.py
    python3 benchmark.py --tokens 1024 --concurrency 1 2 4 8 16 32
    python3 benchmark.py --thinking off      # disable reasoning for A/B
"""
import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")
MODEL = os.environ.get("SERVED_MODEL_NAME", "Qwen3.8-27B")

PROMPT = "Write a detailed technical essay about the history of computing."


def _extra_body(thinking):
    if thinking is None:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": thinking}}


def run_stream(client, max_tokens, thinking):
    """Run one streaming request. Return (ttft_s, wall_s, completion_tokens)."""
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_extra_body(thinking),
    )
    ttft = None
    usage = None
    for ev in stream:
        if getattr(ev, "usage", None) is not None:
            usage = ev.usage
        if ev.choices:
            d = ev.choices[0].delta
            dumped = d.model_dump() if hasattr(d, "model_dump") else {}
            if ttft is None and any(
                v for k, v in dumped.items() if k != "role" and v
            ):
                ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    n = usage.completion_tokens if usage else 0
    return (ttft or wall), wall, n


def bench_single(client, max_tokens, repeats, thinking):
    print(f"\n--- single stream ({max_tokens} output tokens, {repeats} runs, "
          f"thinking={thinking}) ---")
    run_stream(client, 8, thinking)  # warmup
    tpss, ttfts = [], []
    for i in range(repeats):
        ttft, wall, n = run_stream(client, max_tokens, thinking)
        decode = (n - 1) / (wall - ttft) if n > 1 and wall > ttft else 0.0
        tpss.append(decode)
        ttfts.append(ttft)
        print(f"  run {i+1}: ttft={ttft*1000:7.1f} ms  decode={decode:7.2f} tok/s"
              f"  total={n/wall:7.2f} tok/s  tokens={n}")
    med = statistics.median(tpss)
    print(f"  >> median decode = {med:.2f} tok/s | "
          f"median ttft = {statistics.median(ttfts)*1000:.1f} ms")
    return med


def bench_concurrency(client, max_tokens, concurrency, thinking):
    print(f"\n--- concurrency={concurrency} ({concurrency} reqs x {max_tokens} tokens) ---")

    def job(_):
        _, wall, n = run_stream(client, max_tokens, thinking)
        return wall, n

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(job, range(concurrency)))
    wall = time.perf_counter() - t0
    total = sum(n for _, n in results)
    print(f"  wall={wall:.2f}s  out_tokens={total}  aggregate={total/wall:.2f} tok/s")
    return total / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--thinking", choices=["on", "off"], default="off",
                    help="reasoning mode used for the test (default off for a "
                         "stable, apples-to-apples decode-rate number)")
    args = ap.parse_args()
    thinking = {"on": True, "off": False}[args.thinking]

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "EMPTY")
    try:
        print("model:", [m.id for m in client.models.list().data])
    except Exception as e:  # noqa: BLE001
        print("cannot reach server:", e)
        return 1

    best_single = bench_single(client, args.tokens, args.repeats, thinking)
    best_agg = 0.0
    for c in args.concurrency:
        best_agg = max(best_agg, bench_concurrency(client, args.tokens, c, thinking))

    print("\n================ SUMMARY ================")
    print(f"peak single-stream decode : {best_single:.2f} tok/s")
    print(f"peak aggregate throughput : {best_agg:.2f} tok/s")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
