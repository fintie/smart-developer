from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

load_dotenv()

ASYNC_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://smart_dev:smart_dev_password@127.0.0.1:55435/smart_developer",
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_user_tables() -> bool:
    from backend.app.models.users import Base

    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except SQLAlchemyError:
        return False
    return True
