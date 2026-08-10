from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.crud import collections as collection_crud
from backend.app.models.collections import Collection
from backend.app.schemas.collections import CollectionCreate, CollectionResponse


def to_response(collection: Collection) -> CollectionResponse:
    return CollectionResponse(
        id=collection.id,
        rid=collection.rid,
        address=collection.address,
        site=collection.site_data,
        created_at=collection.created_at,
    )


async def list_collections(db: AsyncSession, user_id: int) -> list[CollectionResponse]:
    records = await collection_crud.list_for_user(db, user_id)
    return [to_response(record) for record in records]


async def save_collection(
    db: AsyncSession,
    user_id: int,
    payload: CollectionCreate,
) -> CollectionResponse:
    record = await collection_crud.create_or_get(
        db,
        user_id=user_id,
        rid=payload.rid,
        address=payload.address,
        site_data=payload.site,
    )
    return to_response(record)


async def remove_collection(db: AsyncSession, user_id: int, collection_id: int) -> bool:
    return await collection_crud.delete_for_user(db, user_id, collection_id)
