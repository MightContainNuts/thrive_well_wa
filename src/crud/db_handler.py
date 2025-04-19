import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session, select

from src.models.models import User, Message  # noqa: F401  # Needed for table creation

load_dotenv()
db_url_prod = os.getenv("DATABASE_URL")


class DataBaseHandler:
    """
    This class is responsible for handling the database operations.
    """

    def __init__(self, db_url=db_url_prod):
        """create instance variables"""
        self.engine = None
        self.session = None
        self.db_url = db_url

    def __enter__(self):
        """Initialize the database connection via context manager."""
        self.create_engine()
        self.create_schema()
        self.create_session()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the context manager and dispose of the engine."""
        if self.session:
            self.session.close()

        if self.engine:
            self.engine.dispose()
        if exc_type or exc_val or exc_tb:
            print(f"An error occurred: \n {exc_val} \n {exc_tb} \n {exc_type}")
        print("Database connection closed.")

    def create_engine(self):
        """Create the database engine."""
        self.engine = create_engine(self.db_url, echo=True)

    def create_session(self):
        """Create a session for the database."""
        if self.engine:
            self.session = Session(self.engine)
        else:
            raise ValueError("Engine not created. Call create_engine first.")

    def create_schema(self):
        """create the database schema (tables) if they don't exist."""
        """Create the database schema (tables) if they don't exist."""
        if self.engine:
            SQLModel.metadata.create_all(self.engine)
        else:
            raise ValueError("Engine not created. Call create_engine first.")

    def execute_query(self, query: str):
        """
        Execute a query on the database.
        """
        pass

    def create_new_user(self, message_data: dict) -> User:
        """Create a new user in the database."""
        telegram_id = message_data.get("message", {}).get("from", {}).get("id")
        first_name = (
            message_data.get("message", {}).get("from", {}).get("first_name", "")
        )
        last_name = message_data.get("message", {}).get("from", {}).get("last_name", "")
        user_name = message_data.get("message", {}).get("from", {}).get("username", "")
        is_bot = message_data.get("message", {}).get("from", {}).get("is_bot")

        chat_summary = (
            f"User Created: {telegram_id}: {last_name}, {first_name} - {datetime.now()}"
        )
        print(f"Chat summary: {chat_summary}")
        new_user = User(
            telegram_id=telegram_id,
            user_name=user_name,
            first_name=first_name,
            last_name=last_name,
            is_bot=is_bot,
            chat_summary=chat_summary,
        )
        self.session.add(new_user)
        self.session.commit()
        return new_user

    def get_user(self, message_data: dict, create_if_missing: bool = True) -> User:
        """Get a user by telegram id."""
        telegram_id = message_data.get("message", {}).get("from", {}).get("id")
        print(f"Telegram ID: {telegram_id}")
        user = self.session.exec(
            select(User).where(User.telegram_id == telegram_id)
        ).first()
        if user is None and create_if_missing:
            print(f"User with ID {telegram_id} not found.")
            user = self.create_new_user(message_data)

        else:
            print(f"User found: {user.user_id} {user.telegram_id}")
        return user

    def get_all_users(self) -> list[User]:
        """Get all users from the database."""
        users = self.session.exec(select(User)).all()
        return users

    def save_message(self, message_data: dict):
        """Save a message to the database."""
        telegram_id = message_data.get("message", {}).get("from", {}).get("id")
        message = message_data.get("message", {}).get("text")
        response_entity = (
            "user"
            if not message_data.get("message", {}).get("from", {}).get("is_bot")
            else "bot"
        )
        timestamp = message_data.get("message", {}).get("date")

        print(f"Extracted Telegram ID: {telegram_id}")  # Debugging line
        print(f"Extracted Timestamp: {timestamp}")

        if telegram_id is None:
            raise ValueError("telegram_id is missing in the message data")

        new_message = Message(
            telegram_id=telegram_id,
            message=message,
            response_entity=response_entity,
            timestamp=timestamp,
        )
        self.session.add(new_message)
        self.session.commit()
        print(f"Message saved: {new_message}")

    def save_bot_response(self, response_data: dict):
        """Save a bot response to the database."""
        telegram_id = response_data.get("result", {}).get("from", {}).get("id")
        message = response_data.get("result", {}).get("text")
        response_entity = "bot"
        timestamp = response_data.get("result", {}).get("date")

        new_message = Message(
            telegram_id=telegram_id,
            message=message,
            response_entity=response_entity,
            timestamp=timestamp,
        )
        self.session.add(new_message)
        self.session.commit()
        print(f"Bot response saved: {new_message}")

    def retrieve_messages_by_user(self, user: User) -> list[Message]:
        """Retrieve messages for a given telegram id."""
        telegram_id = user.telegram_id
        results = self.session.exec(
            select(Message).where(Message.telegram_id == telegram_id)
        ).all()

        return results

    def retrieve_message_by_user_timestamp(
        self, user: User, timestamp: int
    ) -> Optional[str]:
        """Retrieve messages for a given telegram id."""
        result = self.session.exec(
            select(Message).where(
                Message.telegram_id == user.telegram_id, Message.timestamp == timestamp
            )
        ).first()
        return result.message if result else None

    def delete_all_user_messages(self, user) -> None:
        """Delete all messages for a given telegram id."""
        messages = self.session.exec(
            select(Message).where(Message.telegram_id == user.telegram_id)
        ).all()
        for message in messages:
            self.session.delete(message)
        self.session.commit()

    def delete_user(self, user) -> None:
        """Delete user from the database."""
        user = self.session.exec(
            select(User).where(User.telegram_id == user.telegram_id)
        ).first()
        if user:
            print(f"Deleting user: {user.first_name} {user.last_name}")
            self.session.delete(user)
            self.session.commit()
        else:
            print(f"User with ID {user.telegram_id} not found.")

    def create_bot_user_account(self):
        """Test the create_bot_user_account method."""

        # Test creating a bot user account
        bot = User(
            telegram_id=8116057140,
            first_name="vita.samaya",
            is_bot=True,
            chat_summary=f"User Created: {8116057140}: vita.samaya - {datetime.now()}",
        )
        self.session.add(bot)
        self.session.commit()
        print(f"Bot user account created: {bot.first_name}")

    def delete_bot_user_account(self):
        """Test the delete_bot_user_account method."""
        bot = self.session.exec(
            select(User).where(User.telegram_id == 8116057140)
        ).first()
        messages = self.session.exec(
            select(Message).where(Message.telegram_id == bot.telegram_id)
        ).all()
        for message in messages:
            self.session.delete(message)

        if bot:
            print(f"Deleting bot user: {bot.first_name}")
            self.session.delete(bot)
            self.session.commit()
        else:
            print(f"Bot user with ID {8116057140} not found.")

        return bot
