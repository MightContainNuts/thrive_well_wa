from fastapi import APIRouter

public_routes = APIRouter()


@public_routes.get("/health", status_code=418)
def read_root(message: str = "I am a teacup"):
    return {"message": message}
