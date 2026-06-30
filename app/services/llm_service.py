"""异步 LLM 服务（DeepSeek，OpenAI 兼容接口）。

基座 ``llm_client.py`` 是同步 ``OpenAI`` 客户端（供 Streamlit 用）。后端改用
``AsyncOpenAI``，使 LLM 调用在 async FastAPI 里全程不阻塞 worker：

* :func:`generate`     —— 一次性返回（意图识别）
* :func:`chat_stream`  —— async generator 流式返回（答案生成，供 SSE）

后续阶段五将在此演进为多模型分层路由（Haiku/DeepSeek 做简单任务、强模型做最终回答）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextvars import ContextVar, Token
from dataclasses import dataclass

from openai import AsyncOpenAI, BadRequestError

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None
_usage_accumulator: ContextVar["TokenUsageAccumulator | None"] = ContextVar(
    "llm_usage_accumulator",
    default=None,
)


@dataclass
class TokenUsageAccumulator:
    """一次业务请求内累计的 LLM token 用量。"""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        return (
            self.input_tokens / 1000 * settings.LLM_INPUT_COST_PER_1K_TOKENS
            + self.output_tokens / 1000 * settings.LLM_OUTPUT_COST_PER_1K_TOKENS
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": round(self.cost, 8),
        }


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL
        )
    return _client


def begin_usage_tracking() -> Token[TokenUsageAccumulator | None]:
    """开始累计当前请求内的 LLM 用量。"""
    return _usage_accumulator.set(TokenUsageAccumulator(model=settings.DEEPSEEK_MODEL))


def end_usage_tracking(token: Token[TokenUsageAccumulator | None]) -> None:
    """结束当前请求的 LLM 用量累计，恢复上层上下文。"""
    _usage_accumulator.reset(token)


def current_usage_snapshot() -> TokenUsageAccumulator:
    """返回当前累计值的副本；没有开启追踪时返回 0。"""
    acc = _usage_accumulator.get()
    if acc is None:
        return TokenUsageAccumulator(model=settings.DEEPSEEK_MODEL)
    return TokenUsageAccumulator(
        model=acc.model,
        input_tokens=acc.input_tokens,
        output_tokens=acc.output_tokens,
    )


def record_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str | None = None,
) -> None:
    """给当前请求累计 token；测试和 API usage 解析共用。"""
    acc = _usage_accumulator.get()
    if acc is None:
        return
    if model:
        acc.model = model
    acc.input_tokens += max(int(input_tokens or 0), 0)
    acc.output_tokens += max(int(output_tokens or 0), 0)


def _record_openai_usage(usage, prompt: str, completion: str = "") -> bool:
    """记录 OpenAI/DeepSeek usage；没有 usage 时返回 False。"""
    if usage is None:
        return False
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if output_tokens == 0 and total_tokens and input_tokens:
        output_tokens = max(total_tokens - input_tokens, 0)
    if input_tokens == 0 and output_tokens == 0:
        return False
    record_usage(input_tokens=input_tokens, output_tokens=output_tokens)
    return True


def _record_estimated_usage(prompt: str, completion: str) -> None:
    """API 未返回 usage 时做保守估算，保证 token_usage 至少有可观测数据。"""
    record_usage(
        input_tokens=_estimate_tokens(prompt),
        output_tokens=_estimate_tokens(completion),
    )


def _estimate_tokens(text: str) -> int:
    """轻量估算 token 数；用于 provider 不返回 usage 的兜底。"""
    if not text:
        return 0
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other_count = max(len(text) - cjk_count, 0)
    return max(1, cjk_count + (other_count + 3) // 4)


async def generate(prompt: str, temperature: float = 0.3) -> str:
    """一次性生成（用于意图识别）。"""
    resp = await _get_client().chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    if not _record_openai_usage(resp.usage, prompt, content):
        _record_estimated_usage(prompt, content)
    return content


async def generate_tool_calls(
    prompt: str,
    tools: list[dict],
    temperature: float = 0.0,
) -> tuple[list[dict], str]:
    """Use OpenAI-compatible native tool calling when the provider supports it.

    Returns ``(tool_calls, content)`` where each tool call is
    ``{"name": str, "arguments": dict, "id": str | None}``.
    """

    resp = await _get_client().chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto",
        stream=False,
        temperature=temperature,
    )
    message = resp.choices[0].message
    content = message.content or ""
    tool_calls: list[dict] = []
    for item in getattr(message, "tool_calls", None) or []:
        function = getattr(item, "function", None)
        if function is None:
            continue
        raw_args = getattr(function, "arguments", "") or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(
            {
                "id": getattr(item, "id", None),
                "name": getattr(function, "name", ""),
                "arguments": arguments,
            }
        )
    completion_text = content + json.dumps(tool_calls, ensure_ascii=False)
    if not _record_openai_usage(resp.usage, prompt, completion_text):
        _record_estimated_usage(prompt, completion_text)
    return tool_calls, content


async def chat_stream(prompt: str, temperature: float = 0.3) -> AsyncIterator[str]:
    """流式生成（用于答案生成）：逐块 yield 文本增量。"""
    try:
        stream = await _get_client().chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=temperature,
            stream_options={"include_usage": True},
        )
    except BadRequestError as exc:
        if "stream_options" not in str(exc):
            raise
        logger.warning("当前 LLM 接口不支持 stream_options.include_usage，降级为估算 token")
        stream = await _get_client().chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=temperature,
        )
    answer_parts: list[str] = []
    usage_recorded = False
    async for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_recorded = (
                _record_openai_usage(chunk.usage, prompt, "".join(answer_parts))
                or usage_recorded
            )
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            answer_parts.append(delta)
            yield delta
    if not usage_recorded:
        _record_estimated_usage(prompt, "".join(answer_parts))
