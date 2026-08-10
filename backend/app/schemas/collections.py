from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    rid: str = Field(min_length=1, max_length=120)
    address: str = Field(min_length=1, max_length=500)
    site: dict[str, Any]


class CollectionResponse(BaseModel):
    id: int
    rid: str
    address: str
    site: dict[str, Any]
    created_at: datetime
