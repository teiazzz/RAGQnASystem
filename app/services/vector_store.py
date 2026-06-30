"""pgvector raw SQL helpers。

避免引入 Python ``pgvector`` 包：向量以字符串字面量 ``[0.1,...]`` 传给 PostgreSQL
的 ``vector`` 类型。若扩展/列不存在，调用方捕获异常后降级。
"""

from __future__ import annotations

import math

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings


def vector_literal(vector: list[float]) -> str:
    """把 float list 转成 pgvector 可 cast 的字面量。"""
    values = []
    for item in vector[: settings.EMBEDDING_DIM]:
        value = float(item)
        if not math.isfinite(value):
            value = 0.0
        values.append(f"{value:.8f}")
    return "[" + ",".join(values) + "]"


async def update_pgvector_embedding(
    session: AsyncSession, chunk_id: int, embedding: list[float]
) -> None:
    """写入可选的 ``embedding_vector`` 列。"""
    await session.execute(
        text(
            "UPDATE document_chunks "
            "SET embedding_vector = CAST(:embedding AS vector) "
            "WHERE id = :chunk_id"
        ),
        {"embedding": vector_literal(embedding), "chunk_id": chunk_id},
    )


async def search_pgvector(
    session: AsyncSession, embedding: list[float], limit: int
) -> list[dict]:
    """按 cosine distance 查询 pgvector top-k。"""
    result = await session.execute(
        text(
            "SELECT id, source_type, source_id, source_title, section, content, "
            "authority_level, meta, "
            "GREATEST(0, 1 - (embedding_vector <=> CAST(:embedding AS vector))) "
            "AS vector_score "
            "FROM document_chunks "
            "WHERE embedding_vector IS NOT NULL "
            "ORDER BY embedding_vector <=> CAST(:embedding AS vector) "
            "LIMIT :limit"
        ),
        {"embedding": vector_literal(embedding), "limit": limit},
    )
    return [dict(row) for row in result.mappings().all()]
