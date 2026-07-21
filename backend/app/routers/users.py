from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from backend.app.config.db_config import get_db
from backend.app.crud import users
from backend.app.models.users import User
from backend.app.schemas.users import UserAuthResponse, UserInfoResponse, UserRequest
from backend.app.utils.auth import get_current_user
from backend.app.utils.security import verify_password

router = APIRouter(prefix="/api/user", tags=["users"])


def success_response(message: str, data):
    return {
        "code": 200,
        "message": message,
        "data": data,
    }


@router.post("/register")
async def register(user_data: UserRequest, db: AsyncSession = Depends(get_db)):
    try:
        existing_user = await users.get_user_by_username(db, user_data.username)
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
        user = await users.create_user(db, user_data)
        token = await users.create_token(db, user.id)
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable. Start PostgreSQL and try again.",
        ) from exc

    return {
        "code": 200,
        "message": "Registration succeeded",
        "data": {
            "token": token,
            "userInfo": {
                "id": user.id,
                "username": user.username,
                "bio": user.bio,
                "avatar": user.avatar,
            }
        },
    }


@router.post("/login")
async def login(user_data: UserRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await users.get_user_by_username(db, user_data.username)
        if not user or not verify_password(user_data.password, user.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

        token = await users.create_token(db, user.id)
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable. Start PostgreSQL and try again.",
        ) from exc

    response_data = UserAuthResponse(
        token=token,
        user_info=UserInfoResponse.model_validate(user),
    )
    return success_response(message="Login succeeded", data=response_data)


# Get the current user from the token.
@router.get("/info")
async def get_user_info(user: User = Depends(get_current_user)):
    return success_response(message="User info retrieved successfully", data=UserInfoResponse.model_validate(user))
