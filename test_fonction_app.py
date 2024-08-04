from fast_API import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Bienvenue dans l'API du projet 7"}


def test_prediction():
    response = client.post("/predict/", json={"id": 97})
    assert response.status_code == 200
    assert response.json() == 0.13584153006616753


def test_check_client_exists():
    response = client.get("/check_client_exists?id=97")
    assert response.status_code == 200
    assert response.json() == True


def test_check_client_does_not_exist():
    response = client.get("/check_client_exists?id=400000")
    assert response.status_code == 200
    assert response.json() == False