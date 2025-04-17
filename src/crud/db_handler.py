from sqlmodel import SQLModel, create_engine, Session, select
from datetime import datetime
from dotenv import load_dotenv
import os

from src.models.models import User, Message  # noqa: F401  # Needed for table creation


class DataBaseHandler:
    """
    This class is responsible for handling the database operations.
    """

    def __init__(self):
        """create instance variables"""
        self.engine = None
        self.session = None

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
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        self.engine = create_engine(db_url, echo=True)
        print(f"Database URL: {db_url}, type: {type(db_url)}")

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
        telegram_id = message_data.get("from", {}).get("id")
        first_name = message_data.get("from", {}).get("first_name", "")
        last_name = message_data.get("from", {}).get("last_name", "")
        is_bot = message_data.get("from", {}).get("is_bot")
        chat_summary = (
            f"User Created: {telegram_id}: {last_name}, {first_name} - {datetime.now()}"
        )
        new_user = User(
            telegram_id=telegram_id,
            user_name=first_name + " " + last_name,
            is_bot=is_bot,
            chat_summary=chat_summary,
        )
        self.session.add(new_user)
        self.session.commit()
        return new_user

    def get_user(self, message_data: dict) -> User:
        """Get a user by telegram id."""
        telegram_id = message_data.get("from", {}).get("id")
        user = self.session.exec(
            select(User).where(User.telegram_id == telegram_id)
        ).first()
        if not user:
            print(f"User with telegram_id {telegram_id} not found.")
            user = self.create_new_user(message_data)
        return user

    def save_message(self, message_data: dict, user: User):
        """Save a message to the database."""
        telegram_id = message_data.get("message", {}).get("from", {}).get("id")
        message = test_data.get("message", {}).get("text")
        response_entity = "user" if not user.is_bot else "bot"
        timestamp = test_data.get("message", {}).get("date")

        new_message = Message(
            telegram_id=telegram_id,
            message=message,
            response_entity=response_entity,
            timestamp=timestamp,
        )
        self.session.add(new_message)
        self.session.commit()
        print(f"Message saved: {new_message}")

    def save_bot_response(self, message_data: dict, user: User):
        """Save a bot response to the database."""
        telegram_id = message_data.get("from", {}).get("id")
        message = message_data.get("message", {}).get("text")
        response_entity = "bot"
        timestamp = message_data.get("date")
        contents = message_data.get("text")

        new_message = Message(
            telegram_id=telegram_id,
            message=message,
            response_entity=response_entity,
            timestamp=timestamp,
            msg_text=contents,
        )
        self.session.add(new_message)
        self.session.commit()
        print(f"Bot response saved: {new_message}")


test_data = {
    "update_id": 292484125,
    "message": {
        "message_id": 17,
        "from": {
            "id": 7594929889,
            "is_bot": False,
            "first_name": "Dean",
            "last_name": "Didion",
            "language_code": "en",
        },
        "chat": {
            "id": 7594929889,
            "first_name": "Dean",
            "last_name": "Didion",
            "type": "private",
        },
        "date": 1744802627,
        "text": "Test",
    },
}

if __name__ == "__main__":
    telegram_id = test_data.get("message", {}).get("from", {}).get("id")
    print(f"Telegram ID: {telegram_id}")
    message = test_data.get("message", {}).get("text")
    print(f"Message: {message}")
    timestamp = test_data.get("message", {}).get("date")
    print(f"Timestamp: {timestamp}")

    # with DataBaseHandler() as db_handler:
    #     db_handler.create_engine()
