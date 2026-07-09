from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    """
    User information table ORM model.
    """

    __tablename__ = "users"

    __table_args__ = (
        Index("username_UNIQUE", "username"),
        Index("phone_UNIQUE", "phone"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="User ID",
    )
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        comment="Username",
    )
    password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Password, stored encrypted",
    )
    nickname: Mapped[Optional[str]] = mapped_column(
        String(50),
        comment="Nickname",
    )
    avatar: Mapped[Optional[str]] = mapped_column(
        String(255),
        comment="Avatar URL",
        default="https://fastly.jsdelivr.net/npm/@vant/assets/cat.jpeg",
    )
    gender: Mapped[Optional[str]] = mapped_column(
        Enum("male", "female", "unknown", name="user_gender"),
        comment="Gender",
        default="unknown",
    )
    bio: Mapped[Optional[str]] = mapped_column(
        String(500),
        comment="Bio",
        default="No bio yet",
    )
    phone: Mapped[Optional[str]] = mapped_column(
        String(20),
        unique=True,
        comment="Phone number",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="Created time",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        comment="Updated time",
    )

    def __repr__(self):
        return (
            f"<User(id={self.id}, "
            f"username='{self.username}', "
            f"nickname='{self.nickname}')>"
        )


class UserToken(Base):
    """
    User token table ORM model.
    """

    __tablename__ = "user_tokens"

    __table_args__ = (
        Index("token_UNIQUE", "token"),
        Index("fk_user_token_user_idx", "user_id"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Token ID",
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        comment="User ID",
    )
    token: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        comment="Token value",
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="Expiration time",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="Created time",
    )

    def __repr__(self):
        return (
            f"<UserToken(id={self.id}, "
            f"user_id={self.user_id}, "
            f"token='{self.token}')>"
        )
