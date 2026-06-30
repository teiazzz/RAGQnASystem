"""回答反馈接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, SessionDep
from app.db.models import Conversation, Feedback, Message
from app.schemas.feedback import FeedbackOut, FeedbackRequest

router = APIRouter(prefix="/feedback", tags=["反馈"])


@router.post("", response_model=FeedbackOut)
async def submit_feedback(
    req: FeedbackRequest,
    user: CurrentUser,
    session: SessionDep,
) -> FeedbackOut:
    """记录当前用户对某条助手回答的赞/踩反馈。"""
    message = await session.scalar(
        select(Message)
        .join(Conversation)
        .where(Message.id == req.message_id, Conversation.user_id == user.id)
    )
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="消息不存在")
    if message.role != "assistant":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只能反馈助手回答",
        )

    comment = (req.comment or "").strip() or None
    feedback = await session.scalar(
        select(Feedback)
        .where(Feedback.message_id == req.message_id, Feedback.user_id == user.id)
        .order_by(Feedback.id.desc())
    )
    if feedback is None:
        feedback = Feedback(
            message_id=req.message_id,
            user_id=user.id,
            rating=req.rating,
            comment=comment,
        )
        session.add(feedback)
    else:
        feedback.rating = req.rating
        feedback.comment = comment

    await session.commit()
    await session.refresh(feedback)
    return FeedbackOut.model_validate(feedback)
