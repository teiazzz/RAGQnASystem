"""应用生命周期：启动时建表 / seed admin / 预热单例，关闭时释放连接池。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.seed import seed_admin
from app.db.session import async_session_maker, engine, init_db
from app.services.bm25_service import get_bm25_service
from app.services.kg_service import get_kg_service
from app.services.ner_service import get_ner_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动：建表 + 初始化管理员……")
    await init_db()
    await seed_admin()
    # 预热单例：NER 初始化较慢（建 AC 自动机 + TF-IDF），提前加载避免首请求阻塞
    logger.info("预热 NER / KG 单例……")
    get_ner_service()
    get_kg_service()
    try:
        async with async_session_maker() as session:
            await get_bm25_service().warmup(session)
        logger.info("BM25 索引预热完成")
    except Exception:
        logger.warning("BM25 索引预热失败，将在首次查询时懒加载", exc_info=True)
    logger.info("启动完成，服务就绪")
    yield
    logger.info("应用关闭：释放数据库连接池")
    await engine.dispose()
