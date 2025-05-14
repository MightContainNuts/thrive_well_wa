from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Column, ForeignKey
from sqlmodel import Field, SQLModel, Relationship
from typing import Optional


class UserRole(str, Enum):
    """Enum for user roles."""

    USER = "user"
    ADMIN = "admin"


class User(SQLModel, table=True):
    """User model for the application."""

    user_id: UUID = Field(default_factory=uuid4, primary_key=True)
    # future use
    user_name: Optional[str] = Field(nullable=True, unique=True)
    first_name: Optional[str] = Field(nullable=True)
    last_name: Optional[str] = Field(nullable=True)
    email: Optional[str] = Field(unique=True, index=True, nullable=True)
    # ------
    created_on: datetime = Field(default_factory=datetime.now)
    updated_on: datetime = Field(default_factory=datetime.now)
    role: UserRole = Field(default=UserRole.USER, nullable=False)
    # telegram info
    telegram_id: int = Field(
        sa_column=Column(
            BigInteger,
            primary_key=True,
        )
    )
    is_bot: bool = Field(default=False, nullable=False)
    chat_summary: str = Field(default=None, nullable=False)
    messages: list["Message"] = Relationship(back_populates="user", cascade_delete=True)


class Message(SQLModel, table=True):
    """Chat summary model for the application."""

    message_id: UUID = Field(default_factory=uuid4, primary_key=True)
    telegram_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("user.telegram_id"),
            index=True,
            unique=False,
        )
    )
    user_query: str = Field(nullable=False)
    ai_response: str = Field(nullable=False)
    evaluation: int = Field(nullable=False)
    timestamp: int = Field(BigInteger, nullable=False)
    user: User = Relationship(back_populates="messages")
