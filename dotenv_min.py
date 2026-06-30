# -*- coding: utf-8 -*-
"""极简 .env 加载器（零第三方依赖）。

在入口 ``login.py`` 最顶部调用 :func:`load`，使 ``config.py`` 与 ``llm_client.py``
能读到 ``.env`` 中的环境变量（DEEPSEEK_API_KEY / NEO4J_PASSWORD 等）。

设计为「不覆盖已存在的真实环境变量」，因此命令行 export 优先级高于 .env。
"""
from __future__ import annotations

import os
from pathlib import Path


def load(path: str = ".env") -> None:
    p = Path(__file__).resolve().parent / path
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)
