"""业务数据表 ORM 定义（SQLAlchemy 2.0 ``Mapped`` 风格）。

5 张表对应改造清单阶段一要求：用户、对话会话、消息、Token 用量、反馈。
- ``messages.entities/intents`` 用 JSONB 存 NER 实体与命中意图，便于管理员调试与后续分析；
- ``token_usage`` 为阶段五成本追踪预留；
- ``feedback`` 为后续 👍/👎 评测预留。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP(0)")
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="新对话")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP(0)")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP(0)"),
        onupdate=text("CURRENT_TIMESTAMP(0)"),
    )

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    entities: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    intents: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    knowledge: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP(0)")
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class TokenUsage(Base):
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    model: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP(0)")
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    rating: Mapped[int] = mapped_column(Integer)  # 1=👍, -1=👎
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP(0)")
    )


class DocumentChunk(Base):
    """RAG 医疗语料切片。

    ``embedding`` 用 JSONB 存一份可移植向量，保证缺少 pgvector 扩展时仍能降级检索。
    启动时若 pgvector 可用，会额外通过 raw SQL 添加 ``embedding_vector vector(N)``
    与 HNSW 索引，检索服务优先走 pgvector。
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_document_chunks_content_hash"),
        Index("ix_document_chunks_source", "source_type", "source_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[str] = mapped_column(String(128), index=True)
    source_title: Mapped[str] = mapped_column(String(255))
    section: Mapped[str] = mapped_column(String(64), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    authority_level: Mapped[str] = mapped_column(String(64), default="medical_corpus")
    meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP(0)")
    )


class UserMemory(Base):
    """用户长期记忆：关键健康事实、过敏史、慢性病、重要医疗决策。

    存储用户特定的医疗背景，按需检索注入 prompt。
    ``embedding`` 字段用于相关性召回过滤（防止新对话被无关记忆污染）。
    ``relevance_keywords`` 用于启发式快速召回（如"糖尿病""青霉素过敏"）。
    """

    __tablename__ = "user_memories"
    __table_args__ = (
        Index("ix_user_memories_user_id_created", "user_id", "created_at"),
        Index("ix_user_memories_category", "category"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(
        String(32), default="health_fact"
    )  # health_fact / allergy / chronic_disease / medication_history / decision
    content: Mapped[str] = mapped_column(Text)
    relevance_keywords: Mapped[list | None] = mapped_column(
        JSONB, nullable=True
    )  # ["糖尿病", "二甲双胍"] 用于快速过滤
    embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    source_conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    importance: Mapped[int] = mapped_column(Integer, default=5)  # 1-10,越高越重要
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP(0)")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP(0)"),
        onupdate=text("CURRENT_TIMESTAMP(0)"),
    )

    user: Mapped["User"] = relationship("User")
