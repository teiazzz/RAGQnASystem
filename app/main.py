"""FastAPI 应用入口。

启动（务必在项目根目录，因 NER 词典/KG 模块用相对路径）::

    uv run uvicorn app.main:app --reload --port 8000

中间件与异常：CORS、X-Request-ID（贯穿日志/响应头）、全局异常兜底为 500 JSON。
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.lifespan import lifespan

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="智能导诊与健康问答助手 API",
    description="基于知识图谱 RAG + 大模型的医疗智能问答后端（阶段一：FastAPI + PostgreSQL）",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """为每个请求生成/透传 X-Request-ID，写入 state 并回填响应头。"""
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# CORS 放在 request_id 之后添加 → 处于最外层，优先处理预检
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底未处理异常（HTTPException/校验错误仍由 FastAPI 默认处理）。"""
    req_id = getattr(request.state, "request_id", None)
    logger.exception("未处理异常 request_id=%s", req_id)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误", "request_id": req_id},
    )


app.include_router(api_router)


@app.get("/", tags=["根"])
async def root() -> dict:
    return {
        "service": "智能导诊与健康问答助手",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
