import os

import requests
from dotenv import load_dotenv
from fastapi import Request, APIRouter

from src.crud.db_handler import DataBaseHandler
from src.services.langgraph_handler import LangGraphHandler

telegram_routes = APIRouter()
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
message_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
lhh = LangGraphHandler()


@telegram_routes.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    telegram_id = data["message"]["from"]["id"]
    user_query = data.get("message", {}).get("text")
    if not user_query or not isinstance(user_query, str):
        response = requests.post(
            message_url,
            json={
                "chat_id": telegram_id,
                "text": "I'm not sure what you mean. Can you please clarify?",
            },
        )
        print(response.status_code, response.text)
        return {"Status": "No message found"}
    print("Incoming Telegram update:", user_query)
    with DataBaseHandler() as db_handler:
        user = db_handler.get_user(
            data
        )  # TODO temp function - look into the /start command
        print(f"DEBUG: User: {user}")
        db_handler.save_message(data)

    lbh = LangGraphHandler(telegram_id=telegram_id)
    ai_response = lbh.chatbot_handler(user_query=user_query)

    response = requests.post(
        message_url, json={"chat_id": telegram_id, "text": ai_response}
    )

    print(f"Status code: {response.status_code} |", ai_response)
    with DataBaseHandler() as db_handler:
        db_handler.save_bot_response(response.json())

    return {
        "Status": response.status_code,
        "chat_id": telegram_id,
        "query": user_query,
        "Response": ai_response,
    }
