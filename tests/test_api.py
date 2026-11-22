from fastapi.testclient import TestClient
from api.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["customers_loaded"] == 1000


def test_valid_prediction():
    with TestClient(app) as client:
        response = client.get("/predict/1001")
        assert response.status_code == 200

        data = response.json()

        assert data["customer_id"] == 1001
        assert 0 <= data["churn_probability"] <= 1
        assert data["predicted_churn"] in [0, 1]
        assert data["model_name"] == "logistic_regression"


def test_customer_not_found():
    with TestClient(app) as client:
        response = client.get("/predict/999999")
        assert response.status_code == 404


def test_invalid_customer_id():
    with TestClient(app) as client:
        response = client.get("/predict/abc")
        assert response.status_code == 422
