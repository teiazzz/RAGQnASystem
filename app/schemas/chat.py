"""聊天与会话相关 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.core.time import format_datetime


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None  # 不传则新建会话


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    entities: dict | None = None
    intents: list | None = None
    sources: list | None = None
    knowledge: str | None = None
    feedback_rating: int | None = None
    feedback_comment: str | None = None
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return format_datetime(value) or ""


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, value: datetime) -> str:
        return format_datetime(value) or ""


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []
