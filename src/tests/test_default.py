from src.main import app, version
from fastapi.testclient import TestClient


api_version = version[0]
api_prefix = f"/api/v{api_version}"
print(f"Testing API version: {api_version}")


def test_read_main():
    client = TestClient(app)
    response = client.get(f"{api_prefix}/health")
    assert response.status_code == 418
    assert response.json() == {"message": "I am a teacup"}
