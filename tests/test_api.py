"""Tests de la API con un modelo falso: no dependen de MLflow ni de haber entrenado."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import ModelInfo

VALID_PAYLOAD = {
    "age": 0.038, "sex": 0.051, "bmi": 0.062, "bp": 0.022, "s1": -0.044,
    "s2": -0.035, "s3": -0.043, "s4": -0.003, "s5": 0.02, "s6": -0.018,
}


class FakeModel:
    def predict(self, data):
        return np.array([123.45])


class BrokenModel:
    def predict(self, data):
        raise RuntimeError("fallo simulado")


FAKE_INFO = ModelInfo(
    model_name="diabetes-regressor",
    model_version="1",
    alias="champion",
    run_id="abc123",
    metadata={"rmse": "56.4504", "model_type": "linear_regression"},
)


@pytest.fixture
def client():
    """Cliente con un modelo falso cargado (sustituye al champion real)."""
    with TestClient(app) as test_client:
        app.state.model = FakeModel()
        app.state.info = FAKE_INFO
        yield test_client


@pytest.fixture
def client_without_model():
    """Cliente con el modelo no disponible (simula que no hay champion)."""
    with TestClient(app) as test_client:
        app.state.model = None
        app.state.info = None
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_returns_metadata(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "diabetes-regressor"
    assert body["model_version"] == "1"
    assert body["alias"] == "champion"
    assert body["metadata"]["rmse"] == "56.4504"


def test_predict_valid_input_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == pytest.approx(123.45)
    assert body["model_version"] == "1"


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "bmi"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_wrong_type_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "age": "abc"}).status_code == 422


def test_predict_out_of_range_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "age": 5}).status_code == 422


def test_predict_extra_field_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "extra": 1}).status_code == 422


def test_predict_without_model_returns_503(client_without_model):
    assert client_without_model.post("/predict", json=VALID_PAYLOAD).status_code == 503


def test_model_info_without_model_returns_503(client_without_model):
    assert client_without_model.get("/model-info").status_code == 503


def test_health_works_even_without_model(client_without_model):
    assert client_without_model.get("/health").status_code == 200


def test_predict_model_failure_returns_500(client):
    app.state.model = BrokenModel()
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 500
