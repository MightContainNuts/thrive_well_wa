from datetime import datetime
import tempfile
from sqlmodel import select, SQLModel

from src.crud.db_handler import DataBaseHandler
from src.models.models import User, Message, ResponseEntity
import pytest


test_data = {
    "update_id": 292484130,
    "message": {
        "message_id": 22,
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
        "date": 1745060257,
        "text": "text testing",
    },
}

bot_test_data = {
    "ok": True,
    "result": {
        "message_id": 26,
        "from": {
            "id": 8116057140,
            "is_bot": True,
            "first_name": "vita.samaya",
            "username": "vita_samaya_bot",
        },
        "chat": {
            "id": 7594929889,
            "first_name": "Dean",
            "last_name": "Didion",
            "type": "private",
        },
        "date": 1745060513,
        "text": "Hey, I received your message!",
    },
}


@pytest.fixture(autouse=True)
def setup_and_teardown():
    with tempfile.NamedTemporaryFile(delete=False) as temp_db_file:
        test_db_url = f"sqlite:///{temp_db_file.name}"
        with DataBaseHandler(test_db_url) as db_handler:
            SQLModel.metadata.create_all(db_handler.engine)
            yield db_handler
            db_handler.session.close()


def test_create_new_user(setup_and_teardown):
    """Test the create_new_user method."""
    db_handler = setup_and_teardown
    # Test the creation of a new user
    user = db_handler.create_new_user(test_data)
    assert user is not None
    assert user.telegram_id == test_data["message"]["from"]["id"]
    assert user.first_name == test_data["message"]["from"]["first_name"]
    assert user.last_name == test_data["message"]["from"]["last_name"]
    assert user.is_bot == test_data["message"]["from"]["is_bot"]


def test_get_user(setup_and_teardown):
    """Test the get_user method."""
    db_handler = setup_and_teardown
    # Test getting an existing user
    db_handler.create_new_user(test_data)
    user = db_handler.get_user(test_data)
    print(user)
    assert user is not None
    assert user.telegram_id == test_data["message"]["from"]["id"]
    assert user.first_name == test_data["message"]["from"]["first_name"]
    assert user.last_name == test_data["message"]["from"]["last_name"]
    assert user.is_bot == test_data["message"]["from"]["is_bot"]


def test_get_all_users(setup_and_teardown):
    """Test the get_all_users method."""
    db_handler = setup_and_teardown
    db_handler.create_new_user(test_data)
    # Test getting all users
    users = db_handler.get_all_users()
    assert users is not None
    assert len(users) >= 1
    for user in users:
        assert isinstance(user.telegram_id, int)
        assert isinstance(user.is_bot, bool)
        assert isinstance(user.created_on, datetime)


def test_save_message(setup_and_teardown):
    """Test the save_message method."""
    db_handler = setup_and_teardown

    # Test saving a message
    db_handler.create_new_user(test_data)
    db_handler.save_message(test_data)
    user = db_handler.get_user(test_data)
    timestamp = test_data["message"]["date"]
    test_message = db_handler.retrieve_message_by_user_timestamp(user, timestamp)
    print(f"{test_message=}")
    assert test_message is not None
    assert test_message == test_data["message"]["text"]


def test_retrieve_messages_by_user(setup_and_teardown):
    """Test the retrieve_messages_by_user_timestamp method."""
    db_handler = setup_and_teardown
    db_handler.create_new_user(test_data)
    # Test retrieving messages by user and timestamp
    user = db_handler.get_user(test_data)
    db_handler.save_message(test_data)

    messages = db_handler.retrieve_messages_by_user(user)
    assert messages is not None
    assert len(messages) >= 1
    for message in messages:
        assert isinstance(message, Message)
        assert isinstance(message.message, str)
        assert isinstance(message.timestamp, int)
        assert isinstance(message.response_entity, ResponseEntity)
        assert message.telegram_id == user.telegram_id


def test_delete_all_user_messages(setup_and_teardown):
    "Test the delete_all_user_messages method."
    db_handler = setup_and_teardown

    # Test deleting all messages for a user
    user = db_handler.get_user(test_data)
    db_handler.delete_all_user_messages(user)
    messages = db_handler.retrieve_messages_by_user(user)
    assert messages is not None
    assert len(messages) == 0


def test_delete_user(setup_and_teardown):
    """Test the delete_user method."""
    db_handler = setup_and_teardown
    db_handler.create_new_user(test_data)

    # Test deleting a user
    user = db_handler.get_user(test_data)
    db_handler.delete_user(user)
    deleted_user = db_handler.session.exec(
        select(User).where(User.telegram_id == user.telegram_id)
    ).first()
    assert deleted_user is None


def test_save_bot_response(setup_and_teardown):
    """Test the save_bot_response method."""
    db_handler = setup_and_teardown

    db_handler.create_bot_user_account()
    # Test saving a bot response
    db_handler.save_bot_response(bot_test_data)
    telegram_id = bot_test_data["result"]["from"]["id"]
    timestamp = bot_test_data["result"]["date"]
    test_message = db_handler.session.exec(
        select(Message).where(
            Message.telegram_id == telegram_id, Message.timestamp == timestamp
        )
    ).first()
    assert test_message is not None
    assert test_message.message == bot_test_data["result"]["text"]
