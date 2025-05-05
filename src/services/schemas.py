from typing import Optional, TypedDict, Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# Telegram schemas for incoming and response messages
class FromModel(BaseModel):
    """From model for incoming messages from Telegram."""

    id: int = Field(examples=[9999999999])
    is_bot: bool = Field(default=True, examples=[False, True])
    first_name: Optional[str] = Field(default=None, examples=["Monty"])
    last_name: Optional[str] = Field(default=None, examples=["Python"])
    user_name: Optional[str] = Field(default=None, examples=["MontyPython"])
    language_code: Optional[str] = Field(default=None, examples=["en", "fr", "es"])


class ChatModel(BaseModel):
    """Chat model for incoming messages from Telegram."""

    id: int = Field(examples=[9999999999])
    first_name: Optional[str] = Field(default=None, examples=["Monty"])
    last_name: Optional[str] = Field(default=None, examples=["Python"])
    type: str = Field(examples=["private", "group", "supergroup", "channel"])


class IncomingMessage(BaseModel):
    """Message model for incoming messages from Telegram."""

    message_id: int = Field(examples=[12345678])
    from_: FromModel = None
    chat: ChatModel = None
    query: str = Field(examples=["Hello, What's the difference between a duck?"])
    date: int = Field(examples=[1714912345])


class ResponseModel(BaseModel):
    """Response model for outgoing messages to Telegram."""

    status: int
    telegram_id: int
    user_query: str
    ai_response: str
    evaluation: str


# Langchain schemas for llm structured outputs
class State(TypedDict):
    """State of the workflow."""

    messages: Annotated[list, add_messages]
    metadata: dict
    telegram_id: int
    summary: str | None


class IsValid(TypedDict):
    """Validation response."""

    is_valid: bool


class EvaluationResponse(TypedDict):
    """Evaluation response."""

    evaluation_success: int
