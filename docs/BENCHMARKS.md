# Benchmarks

All numbers in this document were measured on **4 × Ascend 910B (64 GB)**,
TP=4, greedy decoding (`temperature=0`), thinking disabled, using this
repository's default configuration:

```
--quantization ascend --max-model-len 262144
--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":2}
--additional-config {"ascend_compilation_config":{"fuse_allreduce_rms":true}}
--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY",
                      "cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,24,32,48,64,72,96,128]}
--max-num-seqs 32 --max-num-batched-tokens 8192 --enable-prefix-caching
```

Token counts come from the server's own `usage.completion_tokens`, so reasoning
tokens are included when thinking is enabled.

---

## 1. Single stream

`python3 scripts/benchmark.py --tokens 1024 --concurrency 1 --repeats 5 --thinking off`

```
run 1: ttft= 404.0 ms  decode= 86.86 tok/s  total= 84.06 tok/s  tokens=1024
run 2: ttft= 406.5 ms  decode= 86.31 tok/s  total= 83.53 tok/s  tokens=1024
run 3: ttft= 402.1 ms  decode= 81.75 tok/s  total= 79.28 tok/s  tokens=1024
run 4: ttft= 410.4 ms  decode= 88.30 tok/s  total= 85.37 tok/s  tokens=1024
run 5: ttft= 402.9 ms  decode= 87.03 tok/s  total= 84.22 tok/s  tokens=1024
>> median decode = 86.86 tok/s | median ttft = 404.0 ms
```

A second session with the same configuration measured a median of **83.5 tok/s**
(range 80.8–84.9). Headline figure: **peak 88.3 tok/s, median 83–87 tok/s**.

## 2. MTP speculative decoding acceptance

Counters read from `/metrics` across a deterministic 1800-token generation:

```
spec-decode steps        = 786
draft tokens proposed    = 1572
acceptance @ position 0  = 0.774
acceptance @ position 1  = 0.517
effective tokens / step  = 2.29
```

## 3. Aggregate throughput

`python3 scripts/benchmark_mp.py --concurrency 8 16 32 --tokens 512`

A process-parallel client is used so the client is not the bottleneck.

```
conc=8  : aggregate = 392 tok/s
conc=16 : aggregate = 683 / 696 tok/s
conc=24 : aggregate = 702 tok/s
conc=32 : aggregate = 902 / 1236 / 1189 tok/s
```

**Peak aggregate: ~1 240 tok/s at concurrency 32.**

## 4. Tuning history

| Configuration | Median single-stream decode | Aggregate @ conc 32 |
|---|---|---|
| TP4, MTP-1, power-of-two graph sizes | ~65 tok/s | ~275 tok/s |
| + `num_speculative_tokens=2` | ~83 tok/s | — |
| + `fuse_allreduce_rms` | ~83 tok/s | — |
| + graph sizes aligned to `3 × batch` | ~83 tok/s | **~1 200 tok/s** |
| MTP-3 instead of MTP-2 | 77.6 tok/s | — |

Notes:

* **MTP-2 is optimal.** Three draft positions reduce throughput because the
  extra positions are not accepted often enough to pay for the wider target
  forward pass.
* **Graph-capture granularity matters enormously at high concurrency.** With
  only power-of-two sizes a decode batch of 22–32 requests is padded to a
  128-token graph, wasting up to half of every step. Aligning the sizes to the
  real `3 × batch_size` shapes raised concurrency-32 throughput ~4×.
* `fuse_allreduce_rms` does not change the single-stream number much, but it
  folds the TP all-reduce into the residual-add + RMSNorm and cuts the number
  of HCCL collectives per step.

## 5. Prefix cache behaviour

Reuse granularity is **1536 tokens**; a fixture prompt repeated with identical
content:

| prompt tokens | cached tokens | hit rate | TTFT cold → warm |
|---|---|---|---|
| 676   | 0     | 0 %  | — |
| 3 408 | 1 536 | 45 % | — |
| 3 928 | 1 536 | 39 % | 789 ms → 410 ms |
| 5 226 | 3 072 | 59 % | — |
| 11 732| 9 216 | 79 % | — |

`cached = (floor(prompt_len / 1536) - 1) × 1536`, so there is an effective
**~3072-token minimum prefix** before anything is reused.

## 6. TTFT

| Situation | TTFT |
|---|---|
| prefix-cache hit | 0.42 – 0.55 s |
| cache miss, ~6.8 K prompt | 0.92 s |
| cache miss, ~13.6 K prompt | 1.62 s |
| cache miss, ~27 K prompt | 3.41 s |
| first request after server start | ~5.3 s (one-off warm-up) |
| warm request arriving behind a long prefill | ~7.0 s |

Cold-prefill throughput was measured at **7.4 K – 8.4 K tok/s** with under 1 %
run-to-run variance at a fixed prompt length, which is why a *steady* multi-second
TTFT should be read as "prefix not cached" rather than "server jitter".

## 7. Context and cache size

```
GPU KV cache size: 2,611,412 tokens
Maximum concurrency for 262,144 tokens per request: 9.96x
```

with 8.2 GB of weights per card at TP=4.

---

## Reproducing

```bash
./start_server.sh && ./healthcheck.sh 8000 1800

python3 scripts/smoke_test.py
python3 scripts/test_multimodal.py ./cat.jpg ./demo.mp4
python3 scripts/benchmark.py    --tokens 1024 --concurrency 1 --repeats 5
python3 scripts/benchmark_mp.py --concurrency 8 16 32
```

Host-to-host variation is expected — especially CPU contention and other
tenants on the same machine. Compare like-for-like configurations rather than
absolute numbers.
