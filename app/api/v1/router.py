"""v1 路由汇总。"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, chat, documents, feedback, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(feedback.router)
