import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.inference import BUNDLE_PATH, load_default_employee

client = TestClient(app)


def test_health_always():
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert 'model_loaded' in body


@pytest.mark.skipif(not BUNDLE_PATH.is_file(), reason='python -m src.scripts.export_deploy')
def test_defaults_endpoint():
    response = client.get('/defaults')
    assert response.status_code == 200
    assert 'Age' in response.json()


@pytest.mark.skipif(not BUNDLE_PATH.is_file(), reason='python -m src.scripts.export_deploy')
def test_predict_single():
    payload = load_default_employee()
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body['attrition_probability'] <= 1.0
    assert body['risk_level'] in {'низкий', 'средний', 'высокий'}


@pytest.mark.skipif(not BUNDLE_PATH.is_file(), reason='python -m src.scripts.export_deploy')
def test_predict_batch():
    payload = {'employees': [load_default_employee(), load_default_employee()]}
    response = client.post('/predict/batch', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body['count'] == 2
    assert len(body['predictions']) == 2


@pytest.mark.skipif(not load_default_employee(), reason='нет default_employee.json')
def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setattr('src.api.app.model_is_ready', lambda: False)
    response = client.post('/predict', json=load_default_employee())
    assert response.status_code == 503
