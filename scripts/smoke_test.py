#!/usr/bin/env python3
"""OpenAI-protocol smoke test for the local Qwen3.8-27B-w8a8 vLLM server.

Checks:
  1. /v1/models advertises the model
  2. non-streaming /v1/chat/completions
  3. streaming /v1/chat/completions (SSE)
  4. multi-turn conversation
  5. OpenAI tool-calling (auto tool choice)
"""
import os
import sys
import json
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")
MODEL = os.environ.get("SERVED_MODEL_NAME", "Qwen3.8-27B")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "EMPTY")
ok = True


def section(title):
    print(f"\n=== {title} ===")


def main():
    global ok

    section("1. GET /v1/models")
    models = client.models.list()
    ids = [m.id for m in models.data]
    print("models:", ids)
    if MODEL not in ids:
        print(f"FAIL: {MODEL} not in model list")
        ok = False

    section("2. non-streaming chat.completions")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话介绍你自己。"}],
        max_tokens=128,
        temperature=0.6,
    )
    print("content:", r.choices[0].message.content)
    print("usage:", r.usage)

    section("3. streaming chat.completions")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 10."}],
        max_tokens=128,
        temperature=0.0,
        stream=True,
    )
    chunks = 0
    text = ""
    for ev in stream:
        delta = ev.choices[0].delta.content or ""
        text += delta
        chunks += 1
    print(f"chunks={chunks} text={text[:120]!r}")
    if chunks < 2:
        print("FAIL: streaming produced too few chunks")
        ok = False

    section("4. multi-turn conversation")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Nice to meet you, Ada."},
            {"role": "user", "content": "What is my name?"},
        ],
        max_tokens=64,
        temperature=0.0,
    )
    print("content:", r.choices[0].message.content)

    section("5. tool calling")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "city name"}},
                "required": ["city"],
            },
        },
    }]
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the weather in Beijing? Use the tool."}],
        tools=tools,
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
    )
    msg = r.choices[0].message
    print("finish_reason:", r.choices[0].finish_reason)
    print("tool_calls:", msg.tool_calls)
    if not msg.tool_calls:
        print("WARN: model did not emit a tool call")
    else:
        tc = msg.tool_calls[0]
        print("parsed function:", tc.function.name, tc.function.arguments)
        json.loads(tc.function.arguments)  # must be valid JSON

    print("\n" + ("ALL SMOKE TESTS PASSED" if ok else "SMOKE TESTS HAD FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
