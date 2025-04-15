import logging
from fastapi import Request, APIRouter


telegram_routes = APIRouter()


@telegram_routes.post("/webhook")
async def telegram_webhook(request: Request):
    payload = await request.json()
    logging.info(f"Telegram update: {payload}")
    # handle the message here
    return {"ok": True}
