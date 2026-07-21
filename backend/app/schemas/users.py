from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=72)

    @field_validator("password")
    @classmethod
    def validate_bcrypt_password_length(cls, value: str):
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password cannot be longer than 72 bytes")
        return value


class UserInfoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    bio: str | None = None
    avatar: str | None = None


class UserAuthResponse(BaseModel):
    token: str
    user_info: UserInfoResponse
