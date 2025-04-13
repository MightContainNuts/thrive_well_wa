from src.main import app
from fastapi.testclient import TestClient


def test_default():
    assert True


def test_read_main():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
