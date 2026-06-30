"""鉴权路由：注册 / 登录（OAuth2 表单）/ 当前用户。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.api.deps import CurrentUser, SessionDep
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.schemas.auth import Token, UserOut, UserRegister

router = APIRouter(prefix="/auth", tags=["鉴权"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(data: UserRegister, session: SessionDep) -> User:
    """注册新用户（普通权限）。"""
    existing = await session.scalar(select(User).where(User.username == data.username))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")
    user = User(
        username=data.username,
        password_hash=hash_password(data.password),
        is_admin=False,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
) -> Token:
    """OAuth2 密码模式登录，返回 JWT access_token。"""
    user = await session.scalar(select(User).where(User.username == form.username))
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        user.id, extra={"username": user.username, "is_admin": user.is_admin}
    )
    return Token(access_token=token)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    """返回当前登录用户信息。"""
    return user
