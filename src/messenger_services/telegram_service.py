import os

import requests
from dotenv import load_dotenv
from fastapi import APIRouter


from src.services.langgraph_handler import LangGraphHandler
from src.crud.db_handler import DataBaseHandler as DBHandler
from src.services.schemas import IncomingMessage, ResponseModel


telegram_routes = APIRouter()
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
message_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


@telegram_routes.post(
    "/webhook", summary="Telegram Webhook", response_model=ResponseModel
)
async def telegram_webhook(payload: IncomingMessage):
    """telegram_webhook: Handle incoming messages from Telegram."""
    telegram_id = payload.from_.id
    user_query = payload.query

    # Check if the user exists in the database, create or update as appropriate
    with DBHandler() as db_handler:
        user = db_handler.get_user(payload=payload)
        assert user

    lbh = LangGraphHandler(telegram_id=telegram_id)
    ai_response, evaluation = lbh.chatbot_handler(
        user_query=user_query, timestamp=payload.date
    )

    response = requests.post(
        message_url, json={"chat_id": telegram_id, "text": ai_response}
    )

    return ResponseModel(
        status=response.status_code,
        telegram_id=telegram_id,
        user_query=user_query,
        ai_response=ai_response,
        evaluation=f"{evaluation} %",
    )
