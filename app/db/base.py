"""SQLAlchemy 2.0 声明式基类。

所有 ORM 模型继承 :class:`Base`；``Base.metadata`` 汇总全部表定义，
供 ``init_db`` 的 ``create_all`` 使用。
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
