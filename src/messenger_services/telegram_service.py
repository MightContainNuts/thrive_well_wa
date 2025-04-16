from fastapi import Request, APIRouter
import requests
import os
from dotenv import load_dotenv
from src.crud.db_handler import DataBaseHandler

telegram_routes = APIRouter()
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
message_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


@telegram_routes.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    print("Incoming Telegram update:", data)
    with DataBaseHandler() as db_handler:
        user = db_handler.get_user(data)
        db_handler.save_message(data, user)

    # Optional: auto-reply
    chat_id = data["message"]["chat"]["id"]
    message = "Hey, I received your message!"

    response = requests.post(message_url, json={"chat_id": chat_id, "text": message})
    print(response.status_code, response.text)
    with DataBaseHandler() as db_handler:
        db_handler.save_bot_response(data, user)

    return {"ok": True}
