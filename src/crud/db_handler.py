from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, create_engine, Session, select

from src.models.models import User, Message  # noqa: F401  # Needed for table creation
from src.services.schemas import IncomingMessage, settings


db_url_prod = settings.DATABASE_URL


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
        self.engine = create_engine(self.db_url, echo=False)

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

    def create_new_user(self, payload: IncomingMessage) -> User:
        """Create a new user in the database."""
        telegram_id = payload.from_.id
        first_name = payload.from_.first_name
        last_name = payload.from_.last_name
        user_name = payload.from_.user_name
        is_bot = payload.from_.is_bot

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

    def update_user(self, payload: IncomingMessage) -> User:
        """Update an existing user in the database."""
        telegram_id = payload.from_.id
        first_name = payload.from_.first_name
        last_name = payload.from_.last_name
        user_name = payload.from_.user_name
        is_bot = payload.from_.is_bot

        user = self.session.exec(
            select(User).where(User.telegram_id == telegram_id)
        ).first()
        if user:
            update = False
            if user.first_name != first_name:
                print(f"Updating user: {user.first_name} {user.last_name}")
                user.first_name = first_name
                update = True
            if user.last_name != last_name:
                print(f"Updating user: {user.first_name} {user.last_name}")
                user.last_name = last_name
                update = True
            if user.user_name != user_name:
                print(f"Updating user: {user.first_name} {user.last_name}")
                user.user_name = user_name
                update = True
            if user.is_bot != is_bot:
                print(f"Updating user: is_bot: {is_bot}")
                user.is_bot = is_bot
                update = True
            if update:
                user.chat_summary = f"User Updated: {telegram_id}: {last_name}, {first_name} - {datetime.now()}"
                self.session.commit()
        return user

    def get_user(
        self, payload: IncomingMessage, create_if_missing: bool = True
    ) -> User:
        """Get a user by telegram id."""
        telegram_id = payload.from_.id
        print(f"Telegram ID: {telegram_id}")
        user = self.session.exec(
            select(User).where(User.telegram_id == telegram_id)
        ).first()
        if user is None and create_if_missing:
            print(f"User with ID {telegram_id} not found.")
            user = self.create_new_user(payload)
        else:
            user = self.update_user(payload=payload)
            print(f"User found: {user.user_id} {user.telegram_id}")
        return user

    def get_user_from_id(self, telegram_id: int) -> Optional[User]:
        """Get a user by telegram id."""
        user = self.session.exec(
            select(User).where(User.telegram_id == telegram_id)
        ).first()
        if user is None:
            print(f"User with ID {telegram_id} not found.")
        else:
            print(f"User found: {user.user_id} {user.telegram_id}")
        return user

    def get_all_users(self) -> list[User]:
        """Get all users from the database."""
        users = self.session.exec(select(User)).all()
        return users

    def save_message(
        self,
        telegram_id: int,
        user_query: str,
        ai_response: str,
        evaluation: int,
        timestamp: int,
    ) -> None:
        """Save a message to the database."""

        new_message = Message(
            telegram_id=telegram_id,
            user_query=user_query,
            ai_response=ai_response,
            evaluation=evaluation,
            timestamp=timestamp,
        )
        self.session.add(new_message)
        self.session.commit()
        print(f"Message saved: {new_message}")

    def retrieve_messages_by_user(self, user: User) -> list[Message]:
        """Retrieve messages for a given telegram id."""
        telegram_id = user.telegram_id
        results = self.session.exec(
            select(Message).where(Message.telegram_id == telegram_id)
        ).all()

        return results

    def retrieve_message_by_user_timestamp(
        self, user: User, timestamp: int
    ) -> Optional[Message]:
        """Retrieve messages for a given telegram id."""
        result = self.session.exec(
            select(Message).where(
                (Message.telegram_id == user.telegram_id)
                & (Message.timestamp == timestamp)
            )
        ).first()
        return result if result else None

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
            select(IncomingMessage).where(
                IncomingMessage.telegram_id == bot.telegram_id
            )
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

    def get_chat_summary_from_db(self, telegram_id: int) -> Optional[str | None]:
        """Get chat history from the database."""
        chat_summary = self.session.exec(
            select(User).where(User.telegram_id == telegram_id)
        ).first()

        return chat_summary.chat_summary if chat_summary else None

    def write_chat_summary_to_db(
        self,
        telegram_id: int,
        summary: str,
    ) -> None:
        """Write a chat message to the chat history table."""
        user = self.get_user_from_id(telegram_id)
        if user:
            user.chat_summary = summary
            self.session.commit()
