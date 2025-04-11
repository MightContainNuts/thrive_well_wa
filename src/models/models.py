
from sqlmodel import Field, SQLModel, String, DateTime, JSON, func, Relationship
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime

from pydantic import EmailStr

class UserRole(str, Enum):
    """Enum for user roles."""
    USER = "user"
    ADMIN = "admin"

class User(SQLModel, table=True):
    """User model for the application."""
    user_id: UUID  = Field(default_factory=uuid4, primary_key=True)

    user_name: str = Field(nullable=False, unique=True)
    email: EmailStr = Field(unique=True, index=True, nullable=False)

    created_on: datetime  = Field(default_factory=datetime.now)
    updated_o: datetime = Field(default_factory=datetime.now)
    role: UserRole = Field(default=UserRole.USER, nullable=False)
    chat_summaries: list["ChatSummary"] = Relationship(back_populates="user")




class ChatSummary(SQLModel, table=True):
    """Chat summary model for the application."""
    chat_id: UUID = Field(default_factory=uuid4, primary_key=True)
    summary:str = Field(nullable=False)
    timestamp:datetime = Field(default_factory=datetime.now)
    user_id: UUID = Field(foreign_key="user.user_id")
    user: User = Relationship(back_populates="chat_summaries")
