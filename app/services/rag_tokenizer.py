"""检索用轻量 tokenizer。

医疗问题同时包含中文、英文缩写、药品名和数字。这里用：
- 英文/数字按词切；
- 中文保留单字和 bigram，兼顾召回与专有名词匹配。
"""

from __future__ import annotations

import re

_LATIN_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """把文本切成 BM25/重排可复用的 token。"""
    if not text:
        return []
    tokens = [m.group(0).lower() for m in _LATIN_RE.finditer(text)]
    cjk_chars = _CJK_RE.findall(text)
    tokens.extend(cjk_chars)
    tokens.extend(
        "".join(pair) for pair in zip(cjk_chars, cjk_chars[1:], strict=False)
    )
    return [token for token in tokens if token.strip()]
