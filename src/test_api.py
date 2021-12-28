import requests
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def assert_response(resp_obj, expected_output, status=200):
    assert resp_obj.status_code == status
    response = resp_obj.json()
    assert response == expected_output

def test_root():
    expected_output = {"message": "Hello World"}
    r = client.get("/")
    assert_response(r, expected_output)
    
def test_predictions():
    features = {
        "features": [
            {
            "age": 36,
            "workclass": " State-gov",
            "fnlgt": 210830,
            "education": " Masters",
            "marital-status": " Never-married",
            "occupation": " Prof-specialty",
            "relationship": " Own-child",
            "race": " White",
            "sex": " Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 30
            }
        ]
    }
    expected_output = {
        "predictions": [
            "<=50"
        ]
    }
    r = client.post("/predict/", json=features)
    assert_response(r, expected_output)
    
    
    
    
def test_422_error():
    features = {
        "features": [
            {
                "workclass": " State-gov",
                "fnlgt": 210830,
                "education": " Masters",
                "marital-status": " Never-married",
                "occupation": " Prof-specialty",
                "relationship": " Own-child",
                "race": " White",
                "sex": " Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 30
            }
        ]
    }
    
    expected_output = {
        "detail": [
            {
                "loc": [
                    "body",
                    "features",
                    0,
                    "age"
                ],
                "msg": "field required",
                "type": "value_error.missing"
            }
        ]
    }
    r = client.post("/predict/", json=features)
    assert_response(r, expected_output, status=422)
    
    