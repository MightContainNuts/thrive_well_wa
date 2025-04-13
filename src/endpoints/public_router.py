from fastapi import APIRouter

public_routes = APIRouter()


@public_routes.get("/")
def read_root(message: str = "Hello World"):
    return {"message": message}
