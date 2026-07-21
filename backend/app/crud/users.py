import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.users import User, UserToken
from backend.app.schemas.users import UserRequest
from backend.app.utils import security


async def get_user_by_username(db: AsyncSession, username: str):
    query = select(User).where(User.username == username)
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, user_data: UserRequest):
    hashed_password = security.get_hash_password(user_data.password)
    user = User(username=user_data.username, password=hashed_password)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def create_token(db: AsyncSession, user_id: int):
    token = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(days=7)
    query = select(UserToken).where(UserToken.user_id == user_id)
    result = await db.execute(query)
    user_token = result.scalar_one_or_none()

    if user_token:
        user_token.token = token
        user_token.expires_at = expires_at
    else:
        user_token = UserToken(user_id=user_id, token=token, expires_at=expires_at)
        db.add(user_token)

    await db.commit()
    return token


async def get_user_by_token(db: AsyncSession, token: str):
    query = (
        select(User)
        .join(UserToken, UserToken.user_id == User.id)
        .where(
            UserToken.token == token,
            UserToken.expires_at > datetime.now(),
        )
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()
