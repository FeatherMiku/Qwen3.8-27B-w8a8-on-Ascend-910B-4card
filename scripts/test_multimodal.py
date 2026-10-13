#!/usr/bin/env python3
"""Multimodal (image + video) test for the local Qwen3.8-27B-w8a8 vLLM server.

Uses base64 data URLs so it works with a plain OpenAI client, exactly the way
an external agent harness would call the endpoint.

Usage:
    python3 test_multimodal.py [image] [video]     # defaults: ./cat.jpg ./demo.mp4
    THINKING=1 python3 test_multimodal.py        # keep reasoning enabled
"""
import base64
import mimetypes
import os
import sys
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")
MODEL = os.environ.get("SERVED_MODEL_NAME", "Qwen3.8-27B")
THINKING = os.environ.get("THINKING", "0") == "1"

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "cat.jpg"
VIDEO = sys.argv[2] if len(sys.argv) > 2 else "demo.mp4"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "EMPTY")


def data_url(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def ask(content, max_tokens=1024):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": THINKING}},
    )
    return r.choices[0].message.content, r.usage.completion_tokens


def main():
    print(f"image: {IMAGE}\nvideo: {VIDEO}\nthinking: {THINKING}\n")

    for title, content, mt in [
        ("IMAGE", [
            {"type": "image_url", "image_url": {"url": data_url(IMAGE)}},
            {"type": "text", "text": "Describe this image in detail, including colors and objects."},
        ], 1024),
        ("VIDEO", [
            {"type": "video_url", "video_url": {"url": data_url(VIDEO)}},
            {"type": "text", "text": "Describe what happens in this video."},
        ], 1024),
        ("IMAGE (one-word)", [
            {"type": "image_url", "image_url": {"url": data_url(IMAGE)}},
            {"type": "text", "text": "What is the main animal in this image? Answer with one word."},
        ], 64),
    ]:
        print(f"=== {title} ===")
        text, ntok = ask(content, mt)
        print(text)
        print(f"[completion_tokens={ntok}]\n")


if __name__ == "__main__":
    main()
