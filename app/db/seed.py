"""首次启动数据填充：若 users 表为空，则创建默认管理员 admin/admin123。

替代基座 ``user_data_storage.py`` 的明文 JSON admin，密码改为 bcrypt 哈希存储。
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select

from app.core.security import hash_password
from app.db.models import User
from app.db.session import async_session_maker

logger = logging.getLogger(__name__)


async def seed_admin() -> None:
    async with async_session_maker() as session:
        count = await session.scalar(select(func.count()).select_from(User))
        if count and count > 0:
            return
        admin = User(
            username="admin",
            password_hash=hash_password("admin123"),
            is_admin=True,
        )
        session.add(admin)
        await session.commit()
        logger.info("已初始化默认管理员账户：admin / admin123（请尽快修改密码）")
