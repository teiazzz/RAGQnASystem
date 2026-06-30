"""LLM 客户端：用 DeepSeek API 替换原项目的本地 ollama 调用。

原项目通过 ``ollama.generate`` / ``ollama.chat`` 调本地大模型（qwen:32b 等），
在 4GB 显存机器上无法运行。这里改成调 DeepSeek 的 OpenAI 兼容接口：

* :func:`generate`     —— 一次性返回（用于意图识别）
* :func:`chat_stream`  —— 流式返回（用于答案生成）

环境变量::

    DEEPSEEK_API_KEY   必填，DeepSeek 控制台申请
    DEEPSEEK_MODEL     可选，默认 deepseek-chat

后续 Phase 1 重构为 FastAPI 后端时，这里会演进为统一的多模型路由层
（DeepSeek 做意图识别等简单任务、更强模型做最终医疗回答）。
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from openai import OpenAI

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.deepseek.com"
_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """惰性初始化 OpenAI 客户端（指向 DeepSeek）。"""
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "未检测到 DEEPSEEK_API_KEY 环境变量。请先设置后再启动，例如：\n"
                "  export DEEPSEEK_API_KEY=sk-xxxxxxxx"
            )
        _client = OpenAI(api_key=key, base_url=_BASE_URL)
    return _client


def generate(prompt: str, temperature: float = 0.3) -> str:
    """一次性生成（替换 ``ollama.generate(...)['response']``）。

    用于意图识别：需要拿到完整文本再做正则解析，无需流式。
    """
    resp = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def chat_stream(prompt: str, temperature: float = 0.3) -> Iterator[str]:
    """流式生成（替换 ``ollama.chat(..., stream=True)``）。

    逐块 yield 文本增量，供 Streamlit 实时渲染。
    """
    stream = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=temperature,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
