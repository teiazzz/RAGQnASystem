"""鉴权安全工具：密码哈希（bcrypt）+ JWT 签发/验签（HS256）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """bcrypt 加盐哈希。"""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文与哈希是否匹配。"""
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str | int, extra: dict | None = None) -> str:
    """签发 JWT；``sub`` 为用户 id，过期时间由 ACCESS_TOKEN_EXPIRE_MINUTES 控制。"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    # 注意：python-jose 用标准 json 序列化 claims，日期类 claim 须用 int 时间戳
    payload: dict = {"sub": str(subject), "iat": int(now.timestamp()), "exp": int(expire.timestamp())}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """验签并解析 JWT，过期/篡改会抛 ``jose.JWTError``。"""
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
