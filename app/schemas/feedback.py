"""回答反馈相关 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.core.time import format_datetime


class FeedbackRequest(BaseModel):
    message_id: int = Field(gt=0)
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    message_id: int
    rating: int
    comment: str | None = None
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return format_datetime(value) or ""
