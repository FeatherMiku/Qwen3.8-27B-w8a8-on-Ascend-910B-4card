#!/usr/bin/env python3
"""Aggregate-throughput benchmark using one OS process per request.

The single-process threaded client can become GIL-bound at high concurrency;
this variant uses a process pool so the measured aggregate reflects the server,
not the client.

Usage:
    python3 benchmark_mp.py --concurrency 8 16 32 --tokens 512
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("SERVED_MODEL_NAME", "Qwen3.8-27B")
PROMPT = "Write a detailed technical essay about the history of computing."


def _one(args):
    import time as _t
    from openai import OpenAI
    base_url, model, max_tokens = args
    c = OpenAI(base_url=base_url, api_key="EMPTY")
    t0 = _t.perf_counter()
    s = c.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens, temperature=0.0, stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    n = 0
    for ev in s:
        if getattr(ev, "usage", None):
            n = ev.usage.completion_tokens
    return _t.perf_counter() - t0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    print(f"model={MODEL} base={BASE_URL} tokens={args.tokens}")
    for conc in args.concurrency:
        best = 0.0
        for r in range(args.repeats):
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=conc) as ex:
                res = list(ex.map(_one, [(BASE_URL, MODEL, args.tokens)] * conc))
            wall = time.perf_counter() - t0
            total = sum(n for _, n in res)
            thr = total / wall
            best = max(best, thr)
            print(f"  conc={conc} rep={r+1}: wall={wall:.2f}s tokens={total} "
                  f"aggregate={thr:.2f} tok/s")
        print(f"  >> conc={conc} best aggregate = {best:.2f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
