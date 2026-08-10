from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.collections import Collection


async def list_for_user(db: AsyncSession, user_id: int) -> list[Collection]:
    result = await db.execute(
        select(Collection)
        .where(Collection.user_id == user_id)
        .order_by(Collection.created_at.desc())
    )
    return list(result.scalars().all())


async def create_or_get(
    db: AsyncSession,
    *,
    user_id: int,
    rid: str,
    address: str,
    site_data: dict,
) -> Collection:
    result = await db.execute(
        select(Collection).where(
            Collection.user_id == user_id,
            Collection.rid == rid,
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        return existing

    collection = Collection(
        user_id=user_id,
        rid=rid,
        address=address,
        site_data=site_data,
    )
    db.add(collection)
    await db.commit()
    await db.refresh(collection)
    return collection


async def delete_for_user(db: AsyncSession, user_id: int, collection_id: int) -> bool:
    result = await db.execute(
        delete(Collection).where(
            Collection.id == collection_id,
            Collection.user_id == user_id,
        )
    )
    await db.commit()
    return bool(result.rowcount)
