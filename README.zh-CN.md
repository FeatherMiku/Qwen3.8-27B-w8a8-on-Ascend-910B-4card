# Qwen3.8-27B-w8a8 on Ascend 910B

**面向 [Qwen3.8-27B-w8a8](https://modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8)
的生产级 OpenAI 兼容推理服务，运行于 4 × 昇腾 910B（64 GB），推理框架为
vLLM-Ascend。**

文本 · 图片 · 视频 · 思考推理 · 工具调用 · 256K 上下文 · 流式输出

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Ascend%20910B-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.23.0-blue)
![vLLM--Ascend](https://img.shields.io/badge/vLLM--Ascend-0.23.0.post1-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

> English version: [`README.md`](README.md)

---

## 目录

- [亮点](#亮点)
- [环境要求](#环境要求)
- [获取模型](#获取模型)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 用法](#api-用法)
- [性能实测](#性能实测)
- [设计要点](#设计要点)
- [运维](#运维)
- [容器内运行](#容器内运行)
- [故障排查](#故障排查)
- [仓库结构](#仓库结构)
- [许可证](#许可证)

---

## 亮点

| | |
|---|---|
| **单流解码** | **约 87 tok/s**（峰值 88.3），稳定在 83–87 tok/s |
| **聚合吞吐** | 32 路并发下 **约 1240 tok/s** |
| **上下文** | 原生 **262144 token（256 K）**，可支持约 10 路并发 |
| **多模态** | 支持图片与视频，标准 OpenAI `image_url` / `video_url` |
| **工具调用** | 标准 OpenAI `tools` / `tool_calls`，`finish_reason="tool_calls"` |
| **思考推理** | 3 档强度（`low` / `medium` / `xhigh`），与正文分离返回 |
| **兼容性** | 可直接替换 OpenAI Chat Completions 接口 |

全部由 `start_server.sh` 一个脚本驱动，外加少量 Python 测试/压测工具。
不包含任何框架改动或 fork，就是经过精细调参的 vLLM-Ascend。

---

## 环境要求

### 硬件

* **4 × 昇腾 910B（64 GB）** —— 4 卡张量并行是本仓库实测并调优的配置。
  其他卡数也可运行（见 `TP_SIZE`），但[性能实测](#性能实测)中的数字基于 4 卡。
* 主机需有足够内存加载权重（约 64 GB 峰值即可），以及足够大的 `/dev/shm`
  （引擎启动阶段会使用共享内存做跨进程广播）。

### 软件

使用官方昇腾镜像验证：

```
quay.io/ascend/vllm-ascend:v0.23.0.post1-openeuler
```

| 组件 | 版本 |
|---|---|
| CANN | 9.1.0 |
| vLLM | 0.23.0 |
| vLLM-Ascend | 0.23.0.post1 |
| torch / torch-npu | 2.10.0 / 2.10.0.post4 |
| Python | 3.12 |

> 镜像内已包含全部依赖，**无需额外 `pip install`**。

---

## 获取模型

下载 W8A8 量化权重（约 30 GB）：

```bash
# 方式一：ModelScope CLI
pip install modelscope
modelscope download --model Eco-Tech/Qwen3.8-27B-w8a8 \
  --local_dir ./models/Qwen3.8-27B-w8a8
```

```bash
# 方式二：git lfs
git lfs install
git clone https://www.modelscope.cn/Eco-Tech/Qwen3.8-27B-w8a8.git \
  ./models/Qwen3.8-27B-w8a8
```

之后保持该路径，或用 `MODEL_PATH` 指向别处：

```bash
MODEL_PATH=/data/models/Qwen3.8-27B-w8a8 ./start_server.sh
```

---

## 快速开始

```bash
git clone <this-repo> && cd qwen3.8-27b-w8a8-serving

# 1. 下载权重到 ./models/Qwen3.8-27B-w8a8（见上一节）

# 2. 启动服务（后台守护；写入 logs/ 与 pidfile）
./start_server.sh

# 3. 等待模型加载完成
./healthcheck.sh 8000 1800

# 4. 端到端验证 OpenAI 接口
python3 scripts/smoke_test.py
```

**首次启动需要 5–8 分钟**：加载约 30 GB 权重（TP=4 时每卡约 8.2 GB int8 权重），
并通过 `torch.compile` + ACL Graph 捕获构建解码计算图。后续启动会复用
`~/.cache/vllm/torch_compile_cache`，速度明显更快。

前台运行（Ctrl-C 停止）：

```bash
FOREGROUND=1 ./start_server.sh
```

停止服务：

```bash
./stop_server.sh
```

---

## 配置说明

所有可调项都是 `start_server.sh` 读取的环境变量；设备与运行期环境在 `env.sh`。
开箱即用，无需修改任何文件。

### 服务参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | `<repo>/models/Qwen3.8-27B-w8a8` | 权重目录 |
| `SERVED_MODEL_NAME` | `Qwen3.8-27B` | `/v1/models` 暴露的模型名 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 监听地址 |
| `TP_SIZE` | `4` | 张量并行数（= 使用的 NPU 数） |
| `DP_SIZE` | `1` | 数据并行副本数（保持 1，见[已知问题](#已知问题)） |
| `MAX_MODEL_LEN` | `262144` | 模型原生上限；调小可换更多并发 |
| `GPU_MEM_UTIL` | `0.92` | 引擎可使用的 HBM 比例 |
| `MAX_NUM_SEQS` | `32` | 并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | chunked-prefill 预算 |
| `SPEC_NUM_TOKENS` | `2` | MTP 投机长度；`0` 表示关闭投机推理 |
| `CG_CAPTURE_SIZES` | `[1,2,3,4,6,8,12,16,24,32,48,64,72,96,128]` | ACL Graph 捕获尺寸 |
| `PROMPT_TOKENS_DETAILS` | `1` | 在 `usage` 中返回前缀缓存命中数 |
| `ADDITIONAL_CONFIG` | `{"ascend_compilation_config":{"fuse_allreduce_rms":true}}` | 昇腾融合算子 |
| `ALLOWED_LOCAL_MEDIA_PATH` | *(空 = 关闭)* | **安全敏感项**，见下文 |
| `API_KEY` | *(空 = 不鉴权)* | 设置后要求 `Authorization: Bearer …` |

示例：

```bash
PORT=9000 MAX_MODEL_LEN=131072 ./start_server.sh
SPEC_NUM_TOKENS=0 ./start_server.sh          # 关闭投机推理
API_KEY=sk-...            ./start_server.sh  # 开启 API Key 鉴权
```

### 设备与运行期环境（`env.sh`）

| 变量 | 默认值 | 作用 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3` | 使用的 NPU |
| `VLLM_ASCEND_ENABLE_NZ` | `1` | NZ 权重排布 —— matmul 吞吐最优 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `0` | **必须保持 0**，见[已知问题](#已知问题) |
| `HCCL_BUFFSIZE` | `1024` | HCCL 通信缓冲区（MB） |
| `HCCL_OP_EXPANSION_MODE` | `AIV` | 通信算子下发至 device 侧展开 |
| `TASK_QUEUE_ENABLE` | `2` | 异步 NPU task queue |
| `CPU_AFFINITY_CONF` | `1` | worker 线程绑定 NUMA 本地核 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 减少 HBM 碎片 |

### ⚠️ 安全提示

`--allowed-local-media-path` 会让接口调用方能够读取服务器上任意路径的文件。
本仓库**默认关闭**该能力（上游 vLLM 示例常常默认打开）。确有本地测试需要时，
请显式开启，并尽量把路径收窄：

```bash
ALLOWED_LOCAL_MEDIA_PATH=/srv/media ./start_server.sh
```

推荐用 `data:` base64 URL 或普通 HTTPS URL 传图/传视频。

此外，服务**默认不鉴权**。如果端口暴露在 localhost 之外，请设置 `API_KEY`，
或置于反向代理之后。

---

## API 用法

Base URL `http://<host>:8000/v1`，模型名 `Qwen3.8-27B`。

### Chat Completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "Qwen3.8-27B",
        "messages": [{"role": "user", "content": "用一句话打个招呼。"}],
        "max_tokens": 128,
        "temperature": 0.6
      }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "用一句话打个招呼。"}],
    max_tokens=128,
)
print(r.choices[0].message.content)
```

### 流式输出

```python
stream = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "从 1 数到 10。"}],
    stream=True, max_tokens=128,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### 思考（reasoning）模式

默认**开启思考**。思考内容在 `message.reasoning`，正文在 `message.content`，
两者都会被 `usage.completion_tokens` 计入。支持 3 档：`low`、`medium`、
`xhigh`（默认）—— 其他取值会直接返回 HTTP 400。

```python
# 指定思考强度
client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "证明根号 2 是无理数。"}],
    reasoning_effort="medium",
    max_tokens=2048,
)

# 关闭思考以获得低延迟
client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "把 'good morning' 翻译成法语。"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    max_tokens=64,
)
```

### 工具调用

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
    messages=[{"role": "user", "content": "北京天气如何？请调用工具。"}],
    tools=tools, tool_choice="auto", max_tokens=512,
)
tc = r.choices[0].message.tool_calls[0]
print(tc.function.name, tc.function.arguments)   # get_weather {"city": "Beijing"}
```

`finish_reason` 为 `tool_calls`，且 `arguments` 一定是合法 JSON 字符串。

### 图片与视频

把素材作为 `image_url` / `video_url` content part 传入，HTTPS URL 或 `data:`
base64 都支持。

```python
import base64
b64 = base64.b64encode(open("cat.jpg", "rb").read()).decode()

client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "详细描述这张图片。"},
    ]}],
    max_tokens=512,
)
```

可直接参考 `scripts/test_multimodal.py`（图片 + 视频完整示例）。

### 辅助接口

| 接口 | 用途 |
|---|---|
| `GET /health` | 就绪探针（引擎就绪返回 200） |
| `GET /v1/models` | 模型列表 |
| `GET /metrics` | Prometheus 指标 |

---

## 性能实测

测试环境 **4 × 昇腾 910B（64 GB）**，TP=4，每请求 512–1024 输出 token，
贪心解码，关闭思考，token 数取自服务端 `usage.completion_tokens`。

### 单流

| 指标 | 数值 |
|---|---|
| 峰值解码速率 | **88.3 tok/s** |
| 中位解码速率 | **83 – 87 tok/s** |
| 首 token 时延（TTFT）中位数 | 约 400 ms |
| MTP 接受率 @ 位置 0 / 位置 1 | 0.77 / 0.52 |
| 每个解码 step 有效 token 数 | 2.29 |

### 聚合

| 并发数 | 输出吞吐 |
|---|---|
| 8 | 约 390 tok/s |
| 16 | 约 690 tok/s |
| 32 | **约 1190 – 1240 tok/s** |

### 性能来自哪里

| 步骤 | 效果 |
|---|---|
| 基线（TP4、MTP-1） | 单流约 65 tok/s |
| + `num_speculative_tokens=2` | 约 83 tok/s |
| + `fuse_allreduce_rms` 融合 | 保持约 83 tok/s，同时减少通信次数 |
| + 解码计算图按 `3 × batch` 对齐 | 并发 32 时聚合并吞吐 **提升约 4 倍** |

起决定作用的是两点：**MTP 投机推理**（模型自带 MTP 头，位置 0 接受率 77%、
位置 1 为 52%，把每 step 从约 1 个 token 提升到约 2.29 个）与
**计算图捕获粒度**。

复现命令：

```bash
python3 scripts/benchmark.py    --tokens 1024 --concurrency 1 --repeats 5
python3 scripts/benchmark_mp.py --concurrency 8 16 32
```

完整数据（含上表背后的原始测量记录）见
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)。

---

## 设计要点

这一节记录了若干"不看文档容易踩坑"的点，移植配置前建议先读。

### 这是混合注意力 VLM，不是普通 Transformer

`Qwen3.8-27B` 的架构是 `Qwen3_5ForConditionalGeneration`：64 层里有 48 层是
GDN（gated-delta-net）**线性注意力**，只有 16 层是全注意力。由此：

* 只有 16 层需要 KV Cache，缓存极小 —— 完整 256 K 上下文加上约 10 路并发，
  在每卡约 50 GB 空闲显存下也放得下。
* 开启前缀缓存时，vLLM 会自动选用 Mamba 相关策略（`mamba_cache_mode=align`）。

### MTP 投机推理是最大的单项杠杆

权重自带 1 层 MTP 头，因此 `--speculative-config
'{"method":"qwen3_5_mtp","num_speculative_tokens":2}'` 几乎能让吞吐翻倍。
取 2 是最优值；取 3 反而更慢。

### 计算图捕获尺寸必须与真实解码形状对齐

MTP-2 下每个解码 step 处理 `3 × batch_size` 个 token。如果只捕获 2 的幂次尺寸，
运行时会把 22–32 路并发 padding 到 128 token，单个 step 最多浪费一半算力。
本仓库默认的 `CG_CAPTURE_SIZES` 已按真实倍数对齐。

### 前缀缓存有效，但粒度较粗

复用粒度是 **1536 token**，且存在 **约 3072 token 的最小前缀门槛**；
块哈希是链式的 —— prompt **开头**任何变化都会让整段缓存作废。实践建议：

1. 会话内让 system prompt 与工具定义逐字节稳定。
2. 固定前缀保持 **≥ 3072 token**，否则完全不复用。
3. 不要把长生成任务与对延迟敏感的请求并发执行，长 prefill 会阻塞整个 batch。

| 场景 | TTFT |
|---|---|
| 前缀缓存命中 | 0.42 – 0.55 s |
| 未命中，约 6.8K prompt | 0.92 s |
| 未命中，约 13.6K prompt | 1.62 s |
| 未命中，约 27K prompt | 3.41 s |

冷 prefill 速率约 7.4K–8.4K tok/s，波动 < 1%。所以"稳定地好几秒"说明前缀没命中，
而不是服务在抖动。

### 思考内容在 `message.reasoning`

该 vLLM 版本把思考内容放在 `message.reasoning`（不是 `reasoning_content`），
且 `usage.completion_tokens` 会把思考 token 计入 —— 请相应放大 `max_tokens`。

---

## 运维

### 日志

```bash
tail -f logs/server.log
grep -a "Application startup complete" logs/server.log
```

后台模式会把上一轮日志轮转为 `logs/server.log.1`。

### 健康检查与指标

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/metrics | grep -E "spec_decode|prefix_cache"
```

常用指标：`vllm:spec_decode_num_accepted_tokens_total`、
`vllm:prefix_cache_hits_total`、`vllm:prefix_cache_queries_total`、
`vllm:time_per_output_token_seconds`。

注意 `vllm:prefix_cache_hits_total` 的单位是 **token**，不是请求数。

### 重启

```bash
./stop_server.sh && ./start_server.sh && ./healthcheck.sh 8000 1800
```

`stop_server.sh` 会向整个进程组发信号（vLLM 会派生子进程），
pidfile 缺失时回退到按进程名匹配。

### systemd

`systemd/qwen38-27b.service` 是可直接改用的单元文件 —— 调整
`WorkingDirectory`、`Environment` 与内部路径后：

```bash
sudo cp systemd/qwen38-27b.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen38-27b
```

---

## 容器内运行

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

* `--shm-size` 很重要：引擎启动阶段会通过共享内存做跨进程广播。
* `--device=/dev/davinciN` 会重映射容器内的设备号，NPU 通常显示为 `0,1,2,3`，
  正好匹配默认的 `ASCEND_RT_VISIBLE_DEVICES`。
* 持久化 `~/.cache/vllm/torch_compile_cache` 可以跳过计算图编译。

---

## 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `Address already in use` | 改 `PORT`，或停掉占用该端口的进程。 |
| 启动时报 `expected src to have a size equal to the slice of self` | `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`，改回 `0`。 |
| `tool_calls` 为 `null`，`<tool_call>` XML 出现在 `content` 里 | tool parser 用错了，本模型需要 `qwen3_xml`。 |
| `Unexpected reasoning effort high / minimal`（HTTP 400） | 合法值只有 `low`、`medium`、`xhigh`，请在客户端做映射。 |
| 出现 `max_cudagraph_capture_size ... smaller than ... decode` 告警 | 增大 `CG_CAPTURE_SIZES`，或调小 `MAX_NUM_SEQS` / `SPEC_NUM_TOKENS`。 |
| `content` 为空且 `finish_reason="length"` | `max_tokens` 被思考 token 吃完；调大它，或设 `enable_thinking=false`。 |
| 启动期 OOM | 调小 `GPU_MEM_UTIL`（如 `0.85`）或 `MAX_MODEL_LEN`。 |
| 服务长时间不就绪 | 首次启动编译计算图需 3–6 分钟，观察 `logs/server.log`。 |
| 客户端看不到"缓存命中" | 需 `PROMPT_TOKENS_DETAILS=1`（本仓库默认开启）；同时命中要求共享前缀 ≥ 约 3072 token。 |
| 并发超过约 20 路后聚合吞吐异常低 | 解码 batch 被 padding 到过大计算图；补充 `3 × 并发数` 附近的捕获尺寸。 |

### 已知问题

* **`VLLM_ASCEND_ENABLE_FLASHCOMM1` 必须保持 `0`。** 开启后该稠密混合模型在
  ACL Graph 捕获阶段会失败，报
  `expected src to have a size equal to the slice of self`。
* **数据并行（DP2×TP2）在本模型/本套件上不可用** —— 引擎启动阶段 worker
  初始化失败。请保持 `DP_SIZE=1`。
* **开启投机推理时 `min_p` / `logit_bias` 会被忽略。**
* 思考内容位于 `message.reasoning`，不是 `reasoning_content`。
* Mamba/混合层的前缀缓存上游仍标注为实验特性。本仓库可用，但命中率数字与版本相关。
* 上游镜像更新频繁，需要可复现性时请锁定镜像 tag。

---

## 仓库结构

```
qwen3.8-27b-w8a8-serving/
├── README.md                  # 英文说明
├── README.zh-CN.md            # 本文件
├── LICENSE
├── .gitignore
├── env.sh                     # CANN + 运行期环境
├── start_server.sh            # 启动服务（前台或后台守护）
├── stop_server.sh             # 优雅停止
├── healthcheck.sh             # 等待就绪 + 健康探针
├── systemd/
│   └── qwen38-27b.service     # systemd 单元模板
├── scripts/
│   ├── smoke_test.py          # OpenAI 协议一致性测试
│   ├── test_multimodal.py     # 图片 + 视频测试
│   ├── benchmark.py           # 单流 + 多线程并发压测
│   └── benchmark_mp.py        # 进程级并发聚合压测
└── docs/
    └── BENCHMARKS.md          # 实测结果与调优历程
```

`logs/` 与 `models/` 在运行时生成/使用，已在 `.gitignore` 中忽略。

---


## 许可证

本项目基于 [MIT License](LICENSE) 发布。

**模型权重**按其自身仓库的条款单独分发
（`Eco-Tech/Qwen3.8-27B-w8a8`，基座模型为 Apache-2.0）；本仓库只提供服务脚本。
vLLM 与 vLLM-Ascend 为 Apache-2.0，作为依赖使用。

---

## 致谢

* [vLLM](https://github.com/vllm-project/vllm) 与
  [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)
* [昇腾 / CANN](https://www.hiascend.com/)
* Qwen 团队提供 Qwen3.8-27B 模型及其 ModelSlim 量化工具链
