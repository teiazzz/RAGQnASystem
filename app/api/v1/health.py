"""健康检查：探测 PostgreSQL 与 Neo4j 连通性。"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from sqlalchemy import text

from app.db.session import engine
from app.services.kg_service import get_kg_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["健康检查"])


@router.get("/health")
async def health() -> dict:
    # PostgreSQL
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        logger.exception("PostgreSQL 健康检查失败")
        db_ok = False

    # Neo4j（py2neo 同步，走线程池）
    try:
        neo4j_ok = await asyncio.to_thread(get_kg_service().ping)
    except Exception:
        logger.exception("Neo4j 健康检查失败")
        neo4j_ok = False

    return {
        "status": "ok" if (db_ok and neo4j_ok) else "degraded",
        "database": db_ok,
        "neo4j": neo4j_ok,
    }
