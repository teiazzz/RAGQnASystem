"""异步数据库引擎与会话工厂。

- ``engine``：基于 asyncpg 的异步引擎，带连接池（pool_size=10）与 pre_ping；
- ``async_session_maker``：会话工厂，``expire_on_commit=False`` 便于提交后仍可读对象属性；
- ``get_session``：FastAPI 依赖，每请求一个会话，退出时自动关闭；
- ``init_db``：启动时建表（阶段一用 create_all，Alembic 迁移留作后续）。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.db.base import Base

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    connect_args={"server_settings": {"timezone": settings.APP_TIMEZONE}},
)

async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：提供一个异步会话，请求结束自动关闭。"""
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    """建表（开发期用 metadata.create_all）。"""
    # 触发模型注册到 Base.metadata
    from app.db import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await ensure_pgvector_objects()
    await ensure_message_source_objects()
    await ensure_timestamp_objects()


async def ensure_message_source_objects() -> None:
    """Add message-level RAG source metadata for existing databases."""
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS sources JSONB")
            )
    except Exception:
        logger.warning("messages.sources 初始化失败，历史引用回显可能不可用", exc_info=True)


async def ensure_timestamp_objects() -> None:
    """Use China-local session timezone and second precision for timestamp columns."""
    timestamp_columns = {
        "users": ("created_at",),
        "conversations": ("created_at", "updated_at"),
        "messages": ("created_at",),
        "token_usage": ("created_at",),
        "feedback": ("created_at",),
        "document_chunks": ("created_at",),
    }
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"SET TIME ZONE '{settings.APP_TIMEZONE}'"))
            for table, columns in timestamp_columns.items():
                for column in columns:
                    await conn.execute(
                        text(
                            f"ALTER TABLE {table} "
                            f"ALTER COLUMN {column} TYPE TIMESTAMPTZ(0)"
                        )
                    )
                    await conn.execute(
                        text(
                            f"ALTER TABLE {table} "
                            f"ALTER COLUMN {column} SET DEFAULT CURRENT_TIMESTAMP(0)"
                        )
                    )
    except Exception:
        logger.warning("timestamp 时区/精度初始化失败，继续使用数据库默认配置", exc_info=True)


async def ensure_pgvector_objects() -> None:
    """为 ``document_chunks`` 添加 pgvector 列与 HNSW 索引。

    ORM 表使用 JSONB embedding 保底；这里单独建 vector 列，避免没有 pgvector
    扩展的测试库在 ``create_all`` 阶段失败。
    """
    if not settings.ENABLE_PGVECTOR:
        return
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    "ALTER TABLE document_chunks "
                    f"ADD COLUMN IF NOT EXISTS embedding_vector vector({settings.EMBEDDING_DIM})"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
                    "ON document_chunks USING hnsw "
                    "(embedding_vector vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                )
            )
    except Exception:
        logger.warning(
            "pgvector 初始化失败，已降级为 JSONB embedding + Python 检索",
            exc_info=True,
        )
