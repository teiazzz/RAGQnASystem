"""知识图谱服务（单例）。

持有一个 ``py2neo.Graph`` 连接与基座的 :class:`kg_client.KGClient` 封装。
py2neo 是同步驱动，调用方（chat_service）在线程池中执行查询。连接参数取自 ``app.core.config``。
"""

from __future__ import annotations

import logging

import py2neo

from app.core.config import settings
from app.services.graphrag_service import GraphRAGService
from kg_client import KGClient

logger = logging.getLogger(__name__)


class KGService:
    def __init__(self) -> None:
        self.graph = py2neo.Graph(
            settings.NEO4J_URL,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            name=settings.NEO4J_DBNAME,
        )
        self.client = KGClient(self.graph)
        self.graphrag = GraphRAGService(self.graph)
        logger.info("KG 服务已连接 Neo4j: %s", settings.NEO4J_URL)

    def ping(self) -> bool:
        """健康检查用：执行一条最小查询确认连通。"""
        try:
            self.graph.run("RETURN 1").data()
            return True
        except Exception:
            logger.exception("Neo4j ping 失败")
            return False


_kg_service: KGService | None = None


def get_kg_service() -> KGService:
    """模块级单例。注意 py2neo.Graph 是惰性连接，构造时不会立即建连。"""
    global _kg_service
    if _kg_service is None:
        _kg_service = KGService()
    return _kg_service
