from fastapi import FastAPI

from api.v1.endpoints.public_router import public_routes
from messenger_services.telegram_service import telegram_routes

version = "1.0.0"

app = FastAPI(
    title="ThriveWell",
    description="Telegram / Chatbot API for ThriveWell",
    version=f"{version}",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(public_routes, prefix=f"/api/v{version[0]}", tags=["public"])
app.include_router(
    telegram_routes, prefix=f"/api/v{version[0]}", tags=["messenger services"]
)
