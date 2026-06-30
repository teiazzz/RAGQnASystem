"""聊天路由：SSE 流式问答 + 会话查询。

SSE 事件协议：
* ``event: meta``  —— 一次，含 conversation_id / 实体 / 命中意图 / 知识来源
* ``event: token`` —— 多次，``{"text": "增量"}``
* ``event: done``  —— 一次，``{"message_id": N}``
* ``event: error`` —— 出错时，``{"detail": "..."}``（为阶段五前端错误识别打底）

注意：StreamingResponse 的生成器执行时，路由依赖注入的 DB 会话可能已被释放，
因此生成器内部用 ``async_session_maker()`` 自开独立会话来持久化 assistant 消息。
"""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, SessionDep
from app.core.time import now_app_timezone
from app.db.models import Conversation, Feedback, Message, TokenUsage
from app.db.session import async_session_maker
from app.schemas.chat import ChatRequest, ConversationDetail, ConversationOut
from app.services import chat_service, llm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["对话"])


def _sse(event: str, data: dict) -> str:
    """格式化一个 SSE 事件（data 用 JSON，避免文本换行破坏协议；保留中文）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _persist_token_usage(
    user_id: int,
    conversation_id: int,
    usage: llm_service.TokenUsageAccumulator,
) -> None:
    """把一次 /chat 请求累计的 LLM 用量写入 token_usage。"""
    if usage.total_tokens <= 0:
        return
    async with async_session_maker() as session:
        session.add(
            TokenUsage(
                user_id=user_id,
                conversation_id=conversation_id,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost=usage.cost,
            )
        )
        await session.commit()


@router.post("/chat")
async def chat(req: ChatRequest, user: CurrentUser, session: SessionDep):
    """流式问答。返回 text/event-stream。"""
    # 1. 确定或新建会话（用注入会话做前置写入）
    if req.conversation_id is not None:
        conv = await session.get(Conversation, req.conversation_id)
        if conv is None or conv.user_id != user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    else:
        conv = Conversation(user_id=user.id, title=(req.message[:20] or "新对话"))
        session.add(conv)
        await session.flush()

    # 2. 持久化用户消息
    user_msg = Message(conversation_id=conv.id, role="user", content=req.message)
    session.add(user_msg)
    await session.flush()
    user_msg_id = user_msg.id
    await session.commit()
    conv_id = conv.id
    query = req.message

    # 3. SSE 生成器（内部自开会话）
    async def event_gen():
        usage_token = llm_service.begin_usage_tracking()
        usage_persisted = False

        async def persist_usage_once() -> llm_service.TokenUsageAccumulator:
            nonlocal usage_persisted
            usage = llm_service.current_usage_snapshot()
            if not usage_persisted:
                usage_persisted = True
                await _persist_token_usage(user.id, conv_id, usage)
            return usage

        try:
            result = await chat_service.prepare(
                query,
                conversation_id=conv_id,
                current_message_id=user_msg_id,
                user_id=user.id,  # ✅ 修复：单独传 user_id，用于记忆召回
                user_profile={
                    "user_id": user.id,
                    "role": "admin" if user.is_admin else "user",
                    "is_admin": user.is_admin,
                },
            )
            yield _sse(
                "meta",
                {
                    "conversation_id": conv_id,
                    "entities": result["entities"],
                    "intents": result["intents"],
                    "knowledge": result["knowledge"],
                    "sources": result["sources"],
                    "answer_mode": result.get("answer_mode"),
                    "original_query": result.get("original_query", query),
                    "standalone_query": result.get("standalone_query", query),
                    "rewritten_query": result.get("rewritten_query", query),
                    "multi_queries": result.get("multi_queries", []),
                    "hyde_document": result.get("hyde_document", ""),
                    "conflict_resolution": result.get(
                        "conflict_resolution",
                        {"has_conflict": False, "summary": "", "conflicts": []},
                    ),
                    "agent_tools": result.get(
                        "agent_tools",
                        {
                            "available_tools": [],
                            "planner": "none",
                            "planner_error": "",
                            "orchestrator": "none",
                            "plan": [],
                            "graph_events": [],
                            "iterations": 0,
                            "stop_reason": "",
                            "control": {},
                            "state": {},
                            "final_action": "none",
                            "tool_calls": [],
                            "tool_observations": [],
                        },
                    ),
                },
            )
            answer = ""
            async for delta in chat_service.stream_answer(result["prompt"]):
                answer += delta
                yield _sse("token", {"text": delta})

            # 持久化助手消息（独立会话）
            async with async_session_maker() as s2:
                amsg = Message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=answer,
                    entities=result["entities"],
                    intents=result["intents"],
                    sources=result["sources"],
                    knowledge="\n---\n".join(result["knowledge"]),
                )
                s2.add(amsg)
                conv2 = await s2.get(Conversation, conv_id)
                if conv2 is not None:
                    conv2.updated_at = now_app_timezone()
                await s2.commit()
                await s2.refresh(amsg)
                msg_id = amsg.id
            usage = await persist_usage_once()
            yield _sse(
                "done",
                {
                    "message_id": msg_id,
                    "token_usage": usage.to_dict(),
                },
            )
        except Exception as exc:  # noqa: BLE001 —— 流中异常需转成 error 事件让前端识别
            logger.exception("chat 流式生成失败")
            usage = await persist_usage_once()
            yield _sse(
                "error",
                {
                    "detail": str(exc),
                    "token_usage": usage.to_dict(),
                },
            )
        finally:
            llm_service.end_usage_tracking(usage_token)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(user: CurrentUser, session: SessionDep):
    """列出当前用户的会话（按最近活动倒序）。"""
    rows = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(rows)


@router.get("/conversations/{conv_id}", response_model=ConversationDetail)
async def get_conversation(conv_id: int, user: CurrentUser, session: SessionDep):
    """获取会话详情（含全部消息）。"""
    conv = await session.scalar(
        select(Conversation)
        .where(Conversation.id == conv_id)
        .options(selectinload(Conversation.messages))
    )
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    assistant_message_ids = [
        message.id for message in conv.messages if message.role == "assistant"
    ]
    feedback_by_message_id: dict[int, Feedback] = {}
    if assistant_message_ids:
        rows = await session.scalars(
            select(Feedback)
            .where(
                Feedback.message_id.in_(assistant_message_ids),
                Feedback.user_id == user.id,
            )
            .order_by(Feedback.id.desc())
        )
        for feedback in rows:
            feedback_by_message_id.setdefault(feedback.message_id, feedback)

    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "entities": message.entities,
                "intents": message.intents,
                "sources": message.sources or sources_from_knowledge(message.knowledge),
                "knowledge": message.knowledge,
                "created_at": message.created_at,
                "feedback_rating": (
                    feedback_by_message_id[message.id].rating
                    if message.id in feedback_by_message_id
                    else None
                ),
                "feedback_comment": (
                    feedback_by_message_id[message.id].comment
                    if message.id in feedback_by_message_id
                    else None
                ),
            }
            for message in conv.messages
        ],
    }


def sources_from_knowledge(knowledge: str | None) -> list[dict]:
    """Best-effort source recovery for messages saved before messages.sources existed."""
    if not knowledge:
        return []
    sources: list[dict] = []
    for block in re.split(r"\n---\n", knowledge):
        text = block.strip()
        match = re.match(r"^\[(\d+)\]\s*(.*)$", text, flags=re.S)
        if not match:
            continue
        citation_id = int(match.group(1))
        content = match.group(2).strip()
        sources.append(
            {
                "citation_id": citation_id,
                "chunk_id": None,
                "source_type": "history",
                "source_title": "历史引用来源",
                "section": f"引用 {citation_id}",
                "authority_level": "historical_rag_source",
                "score": 0,
                "vector_score": 0,
                "bm25_score": 0,
                "kg_score": 0,
                "content_preview": content[:180],
                "content": content,
                "metadata": {},
            }
        )
    return sources
