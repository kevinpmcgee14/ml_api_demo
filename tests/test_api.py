import requests
import pytest


root_url = 'http://127.0.0.1:8000/'


def assert_response(resp_obj, expected_output, status=200):
    assert resp_obj.status_code == status
    response = resp_obj.json()
    assert response == expected_output

def test_root():
    expected_output = {"message": "Hello World"}
    r = requests.get(root_url)
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
    url = root_url + 'predict/'
    r = requests.post(url, json=features)
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
    url = root_url + 'predict/'
    r = requests.post(url, json=features)
    assert_response(r, expected_output, status=422)
    
    