# Qwen3.8-27B-w8a8 on Ascend 910B

**Production-ready, OpenAI-compatible inference service for
[Qwen3.8-27B-w8a8](https://modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8)
running on 4 × Ascend 910B (64 GB) with vLLM-Ascend.**

Text · Image · Video · Reasoning · Tool calling · 256K context · Streaming

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Ascend%20910B-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.23.0-blue)
![vLLM--Ascend](https://img.shields.io/badge/vLLM--Ascend-0.23.0.post1-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

---

## Table of contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Getting the model](#getting-the-model)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API usage](#api-usage)
- [Performance](#performance)
- [Design notes](#design-notes)
- [Operations](#operations)
- [Running in a container](#running-in-a-container)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [License](#license)

---

## Highlights

| | |
|---|---|
| **Single-stream decode** | **≈ 87 tok/s** (peak 88.3), stable 83–87 tok/s |
| **Aggregate throughput** | **≈ 1 240 tok/s** at 32 concurrent requests |
| **Context window** | **262 144 tokens** (native) with ~10× concurrency |
| **Multimodal** | image + video input, OpenAI `image_url` / `video_url` parts |
| **Tool calling** | OpenAI `tools` / `tool_calls`, `finish_reason="tool_calls"` |
| **Reasoning** | 3 effort levels (`low` / `medium` / `xhigh`), returned separately |
| **Compatibility** | drop-in OpenAI Chat Completions endpoint |

Everything is driven by one script (`start_server.sh`) plus a small set of
Python test/benchmark utilities. No framework code, no forks — it is plain
vLLM-Ascend with a carefully tuned argument set.

---

## Requirements

### Hardware

* **4 × Ascend 910B (64 GB)** — 4 cards with tensor parallelism is the profiled
  configuration. Other counts are possible (see `TP_SIZE`), but the numbers in
  [Performance](#performance) are for 4.
* Host with enough RAM for weight loading (≈ 64 GB peak is comfortable) and a
  reasonably sized `/dev/shm` (the engine uses shared memory during startup).

### Software

Tested with the official Ascend container image:

```
quay.io/ascend/vllm-ascend:v0.23.0.post1-openeuler
```

| Component | Version |
|---|---|
| CANN | 9.1.0 |
| vLLM | 0.23.0 |
| vLLM-Ascend | 0.23.0.post1 |
| torch / torch-npu | 2.10.0 / 2.10.0.post4 |
| Python | 3.12 |

> No extra `pip install` is required — the image already ships everything.

---

## Getting the model

Download the W8A8 quantized weights (~30 GB):

```bash
# ModelScope CLI
pip install modelscope
modelscope download --model Eco-Tech/Qwen3.8-27B-w8a8 \
  --local_dir ./models/Qwen3.8-27B-w8a8
```

Or use `git lfs`:

```bash
git lfs install
git clone https://www.modelscope.cn/Eco-Tech/Qwen3.8-27B-w8a8.git \
  ./models/Qwen3.8-27B-w8a8
```

Then either keep that path or point `MODEL_PATH` at it:

```bash
MODEL_PATH=/data/models/Qwen3.8-27B-w8a8 ./start_server.sh
```

---

## Quick start

```bash
git clone <this-repo> && cd qwen3.8-27b-w8a8-serving

# 1. weights (see "Getting the model" above) -> ./models/Qwen3.8-27B-w8a8

# 2. start the server (daemonised; logs/ + pidfile)
./start_server.sh

# 3. wait until the model is loaded
./healthcheck.sh 8000 1800

# 4. verify the OpenAI API end to end
python3 scripts/smoke_test.py
```

The **first launch takes 5–8 minutes**: ~30 GB of weights are loaded
(~8.2 GB of int8 weights per card at TP=4) and the decode graphs are built with
`torch.compile` + ACL-graph capture. Later launches reuse
`~/.cache/vllm/torch_compile_cache` and are faster.

Run it attached instead (Ctrl-C stops it):

```bash
FOREGROUND=1 ./start_server.sh
```

Stop it:

```bash
./stop_server.sh
```

---

## Configuration

All tunables are environment variables read by `start_server.sh`; the device and
runtime environment lives in `env.sh`. Nothing needs editing to get started.

### Server flags

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `<repo>/models/Qwen3.8-27B-w8a8` | weight directory |
| `SERVED_MODEL_NAME` | `Qwen3.8-27B` | name exposed by `/v1/models` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | listen address |
| `TP_SIZE` | `4` | tensor parallel size (= number of NPUs) |
| `DP_SIZE` | `1` | data parallel replicas (keep at 1, see [Known issues](#known-issues)) |
| `MAX_MODEL_LEN` | `262144` | native max; lower it for more concurrency |
| `GPU_MEM_UTIL` | `0.92` | fraction of HBM the engine may use |
| `MAX_NUM_SEQS` | `32` | concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | chunked-prefill budget |
| `SPEC_NUM_TOKENS` | `2` | MTP draft length; `0` disables speculative decoding |
| `CG_CAPTURE_SIZES` | `[1,2,3,4,6,8,12,16,24,32,48,64,72,96,128]` | ACL-graph batch sizes |
| `PROMPT_TOKENS_DETAILS` | `1` | report prefix-cache hits in `usage` |
| `ADDITIONAL_CONFIG` | `{"ascend_compilation_config":{"fuse_allreduce_rms":true}}` | Ascend fusion passes |
| `ALLOWED_LOCAL_MEDIA_PATH` | *(empty = disabled)* | **security-sensitive**, see below |
| `API_KEY` | *(empty = no auth)* | set to require `Authorization: Bearer …` |

Examples:

```bash
PORT=9000 MAX_MODEL_LEN=131072 ./start_server.sh
SPEC_NUM_TOKENS=0 ./start_server.sh          # disable speculative decoding
API_KEY=sk-...            ./start_server.sh  # require an API key
```

### Device & runtime environment (`env.sh`)

| Variable | Default | Purpose |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3` | NPUs to serve on |
| `VLLM_ASCEND_ENABLE_NZ` | `1` | NZ weight layout — best matmul throughput |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `0` | **must stay 0**, see [Known issues](#known-issues) |
| `HCCL_BUFFSIZE` | `1024` | HCCL collective buffer size (MB) |
| `HCCL_OP_EXPANSION_MODE` | `AIV` | device-side collective expansion |
| `TASK_QUEUE_ENABLE` | `2` | asynchronous NPU task queue |
| `CPU_AFFINITY_CONF` | `1` | bind worker threads to NUMA-local cores |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | reduce HBM fragmentation |

### ⚠️ Security note

`--allowed-local-media-path` lets API callers make the server read arbitrary
files from its filesystem. It is **disabled by default** here (upstream vLLM
examples often enable it). If you need it for local testing, opt in explicitly
and keep the path as narrow as possible:

```bash
ALLOWED_LOCAL_MEDIA_PATH=/srv/media ./start_server.sh
```

Prefer sending images/videos as `data:` base64 URLs or plain HTTPS URLs.

By default the endpoint is also **unauthenticated**. If it is reachable beyond
localhost, set `API_KEY`, or put it behind a reverse proxy.

---

## API usage

Base URL `http://<host>:8000/v1`, model name `Qwen3.8-27B`.

### Chat completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "Qwen3.8-27B",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 128,
        "temperature": 0.6
      }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=128,
)
print(r.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "Count from 1 to 10."}],
    stream=True, max_tokens=128,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Reasoning (thinking) mode

Thinking is **on by default**. Reasoning tokens are returned in
`message.reasoning`, the answer in `message.content`, and both are counted by
`usage.completion_tokens`. Three effort levels are supported: `low`, `medium`,
`xhigh` (default) — anything else is rejected with HTTP 400.

```python
# explicit effort
client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    reasoning_effort="medium",
    max_tokens=2048,
)

# turn thinking off for low latency
client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "Translate 'good morning' to French."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    max_tokens=64,
)
```

### Tool calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]
r = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "What is the weather in Beijing? Use the tool."}],
    tools=tools, tool_choice="auto", max_tokens=512,
)
tc = r.choices[0].message.tool_calls[0]
print(tc.function.name, tc.function.arguments)   # get_weather {"city": "Beijing"}
```

`finish_reason` is `tool_calls` and `arguments` is always a valid JSON string.

### Images and video

Pass media as `image_url` / `video_url` content parts — HTTPS URLs or `data:`
base64 both work.

```python
import base64
b64 = base64.b64encode(open("cat.jpg", "rb").read()).decode()

client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "Describe this image in detail."},
    ]}],
    max_tokens=512,
)
```

See `scripts/test_multimodal.py` for a runnable image + video example.

### Helper endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness probe (200 when the engine is ready) |
| `GET /v1/models` | model listing |
| `GET /metrics` | Prometheus metrics |

---

## Performance

Measured on **4 × Ascend 910B (64 GB)**, TP=4, 512–1024 output tokens per
request, greedy decoding, thinking disabled, token counts taken from the
server's own `usage.completion_tokens`.

### Single stream

| Metric | Value |
|---|---|
| Peak decode throughput | **88.3 tok/s** |
| Median decode throughput | **83 – 87 tok/s** |
| Median time-to-first-token | ~400 ms |
| MTP acceptance @ pos 0 / pos 1 | 0.77 / 0.52 |
| Effective tokens per decode step | 2.29 |

### Aggregate

| Concurrency | Output throughput |
|---|---|
| 8 | ~390 tok/s |
| 16 | ~690 tok/s |
| 32 | **~1 190 – 1 240 tok/s** |

### What produces these numbers

| Step | Effect |
|---|---|
| Baseline (TP4, MTP-1) | ~65 tok/s single stream |
| + `num_speculative_tokens=2` | ~83 tok/s |
| + `fuse_allreduce_rms` fusion | keeps ~83 tok/s while cutting collectives |
| + decode graphs aligned to `3 × batch` shapes | **+4× aggregate** at concurrency 32 |

Two levers dominate: **MTP speculative decoding** (the model ships an MTP head
that accepts 77 % of drafts at position 0 and 52 % at position 1, lifting the
model from ~1 to ~2.29 tokens per step) and **graph-capture granularity**.

Reproduce with:

```bash
python3 scripts/benchmark.py    --tokens 1024 --concurrency 1 --repeats 5
python3 scripts/benchmark_mp.py --concurrency 8 16 32
```

Full results, including the raw runs behind the table above, are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Design notes

A few non-obvious things this repository gets right. If you are porting the
setup, read this section first.

### The model is a hybrid-attention VLM, not a plain transformer

`Qwen3.8-27B` is `Qwen3_5ForConditionalGeneration`: 64 layers of which 48 are
gated-delta-net (GDN) **linear-attention** layers and only 16 are full
attention. Consequences:

* Only 16 layers hold a KV cache, so the cache is tiny — the full 256K context
  fits with ~10× concurrency in ~50 GB of free HBM per card.
* vLLM auto-selects the Mamba-aware cache strategy
  (`mamba_cache_mode=align`) when prefix caching is enabled.

### MTP speculative decoding is the biggest single lever

The checkpoint ships a 1-layer MTP head, so `--speculative-config
'{"method":"qwen3_5_mtp","num_speculative_tokens":2}'` nearly doubles
throughput. Two is optimal; three is *slower*.

### Graph-capture sizes must be aligned to the real decode shapes

With MTP-2 a decode step processes `3 × batch_size` tokens. Capturing only
power-of-two sizes makes the runtime pad e.g. 22–32 requests up to 128 tokens,
wasting up to half of every step. The default `CG_CAPTURE_SIZES` is aligned to
the actual multiples instead.

### Prefix caching is real, but coarse

Reuse granularity is **1536 tokens** with a **~3072-token minimum prefix**, and
block hashes are chained — changing anything at the *start* of the prompt
invalidates the whole cached prefix. Practical guidance:

1. Keep the system prompt and tool definitions byte-stable across turns.
2. Keep the fixed prefix **≥ 3072 tokens**, otherwise nothing is reused.
3. Do not run auxiliary long generations concurrently with latency-sensitive
   requests; a long prefill blocks the batch.

| Situation | TTFT |
|---|---|
| prefix-cache hit | 0.42 – 0.55 s |
| miss, ~6.8K prompt | 0.92 s |
| miss, ~13.6K prompt | 1.62 s |
| miss, ~27K prompt | 3.41 s |

Cold-prefill throughput is ~7.4K–8.4K tok/s with < 1 % variance, so a *steady*
multi-second TTFT means the prefix is missing, not that the server is jittery.

### `message.reasoning`, not `reasoning_content`

This vLLM build returns reasoning in `message.reasoning`, and
`usage.completion_tokens` includes it — budget `max_tokens` accordingly.

---

## Operations

### Logs

```bash
tail -f logs/server.log
grep -a "Application startup complete" logs/server.log
```

The daemon mode rotates the previous run to `logs/server.log.1`.

### Health and metrics

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/metrics | grep -E "spec_decode|prefix_cache"
```

Useful series: `vllm:spec_decode_num_accepted_tokens_total`,
`vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total`,
`vllm:time_per_output_token_seconds`.

`vllm:prefix_cache_hits_total` counts **tokens**, not requests.

### Restart

```bash
./stop_server.sh && ./start_server.sh && ./healthcheck.sh 8000 1800
```

`stop_server.sh` signals the whole process group (vLLM spawns worker
subprocesses) and falls back to a pattern match if the pidfile is missing.

### systemd

`systemd/qwen38-27b.service` is a ready-to-edit unit — adjust
`WorkingDirectory`, `Environment` and the paths inside, then:

```bash
sudo cp systemd/qwen38-27b.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen38-27b
```

---

## Running in a container

```bash
docker run -d --name qwen38-27b \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v "$PWD":/workspace/qwen3.8-27b-w8a8-serving \
  -p 8000:8000 \
  --shm-size=64g \
  --ipc=host \
  quay.io/ascend/vllm-ascend:v0.23.0.post1-openeuler \
  sleep infinity

docker exec -it qwen38-27b bash
cd /workspace/qwen3.8-27b-w8a8-serving
./start_server.sh && ./healthcheck.sh 8000 1800
```

* `--shm-size` matters: the engine broadcasts across processes via shared
  memory during startup.
* `--device=/dev/davinciN` remaps device ids inside the container, so the NPUs
  usually appear as `0,1,2,3` — which matches the default
  `ASCEND_RT_VISIBLE_DEVICES`.
* Persist `~/.cache/vllm/torch_compile_cache` between runs to skip graph
  compilation.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Address already in use` | Change `PORT`, or stop the process holding it. |
| `expected src to have a size equal to the slice of self` during startup | `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`; set it back to `0`. |
| `tool_calls` is `null` and raw `<tool_call>` XML shows up in `content` | Wrong tool parser; this model needs `qwen3_xml`. |
| `Unexpected reasoning effort high / minimal` (HTTP 400) | Only `low`, `medium`, `xhigh` are valid. Map your client's tiers accordingly. |
| `max_cudagraph_capture_size ... smaller than ... decode` warning | Raise `CG_CAPTURE_SIZES` or lower `MAX_NUM_SEQS` / `SPEC_NUM_TOKENS`. |
| Empty `content` with `finish_reason="length"` | Thinking consumed the whole `max_tokens` budget; raise it or set `enable_thinking=false`. |
| OOM during startup | Lower `GPU_MEM_UTIL` (e.g. `0.85`) or `MAX_MODEL_LEN`. |
| Server never becomes ready | First run compiles graphs for 3–6 minutes — watch `logs/server.log`. |
| Client cannot show a cache-hit indicator | `usage.prompt_tokens_details` is `null` unless `PROMPT_TOKENS_DETAILS=1` (the default here). Also note hits need a ≥ ~3072-token shared prefix. |
| Low aggregate throughput above ~20 concurrent requests | Decode batches are being padded to an oversized graph; add capture sizes near `3 × concurrency`. |

### Known issues

* **`VLLM_ASCEND_ENABLE_FLASHCOMM1` must stay `0`.** With it enabled, ACL-graph
  capture of this dense hybrid model fails with
  `expected src to have a size equal to the slice of self`.
* **Data parallel (DP2×TP2) does not work** with this model/stack — worker
  initialisation fails during engine start. Keep `DP_SIZE=1`.
* **`min_p` / `logit_bias` are ignored** while speculative decoding is enabled.
* Reasoning lives in `message.reasoning`, not `reasoning_content`.
* Prefix caching for Mamba/hybrid layers is flagged experimental upstream. It
  works here, but treat the hit-rate numbers as version-specific.
* The upstream image is a large moving target; pin the tag if you need
  reproducibility.

---

## Repository layout

```
qwen3.8-27b-w8a8-serving/
├── README.md                  # this file
├── README.zh-CN.md            # 中文说明
├── LICENSE
├── .gitignore
├── env.sh                     # CANN + runtime environment
├── start_server.sh            # launch the service (foreground or daemon)
├── stop_server.sh             # graceful stop
├── healthcheck.sh             # wait-for-ready + readiness probe
├── systemd/
│   └── qwen38-27b.service     # systemd unit template
├── scripts/
│   ├── smoke_test.py          # OpenAI protocol conformance test
│   ├── test_multimodal.py     # image + video test
│   ├── benchmark.py           # single-stream + threaded concurrency sweep
│   └── benchmark_mp.py        # aggregate sweep, process-parallel client
└── docs/
    └── BENCHMARKS.md          # measured results and tuning history
```

`logs/` and `models/` are created/expected at runtime and are git-ignored.


---

## License

Released under the [MIT License](LICENSE).

The **model weights** are distributed separately under the terms of their own
repository (`Eco-Tech/Qwen3.8-27B-w8a8`, Apache-2.0 for the base model); this
project only provides serving scripts. vLLM and vLLM-Ascend are Apache-2.0 and
are used as dependencies.

---

## Acknowledgements

* [vLLM](https://github.com/vllm-project/vllm) and
  [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)
* [Ascend / CANN](https://www.hiascend.com/)
* The Qwen team for the Qwen3.8-27B model and its ModelSlim quantization
  toolchain
