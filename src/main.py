from fastapi import FastAPI

from src.endpoints.public_router import public_routes


app = FastAPI()
app.include_router(public_routes)
