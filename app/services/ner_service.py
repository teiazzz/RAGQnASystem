"""规则版 NER 服务（单例）。

复用基座 ``rule_ner.py`` 的 ``rule_find``（AC 自动机）与 ``tfidf_alignment``（TF-IDF 对齐）。
这两个资源初始化开销大（读数万实体建自动机 + 算 TF-IDF 矩阵），但初始化后只读、
查询不改自身状态，因此**进程内初始化一次、跨请求并发复用**（等价于原 Streamlit 的
``@st.cache_resource``）。实体抽取是 CPU 密集的同步操作，用 ``asyncio.to_thread`` 丢线程池，
避免阻塞事件循环。

依赖词典文件 ``data/ent_aug/*.txt``（相对路径），故 **必须从项目根目录启动服务**。
"""

from __future__ import annotations

import asyncio
import logging

import rule_ner

logger = logging.getLogger(__name__)


class NERService:
    def __init__(self) -> None:
        logger.info("初始化规则版 NER 资源（AC 自动机 + TF-IDF 对齐）……")
        self._rule = rule_ner.rule_find()
        self._tfidf = rule_ner.tfidf_alignment()
        logger.info("NER 资源就绪")

    async def extract(self, text: str) -> dict[str, str]:
        """抽取实体，返回 ``{实体类型: 标准实体名}``。CPU 密集，走线程池。"""
        return await asyncio.to_thread(
            rule_ner.get_ner_result, text, self._rule, self._tfidf
        )


_ner_service: NERService | None = None


def get_ner_service() -> NERService:
    """模块级单例；首次调用触发初始化（建议在 lifespan 启动时预热）。"""
    global _ner_service
    if _ner_service is None:
        _ner_service = NERService()
    return _ner_service
