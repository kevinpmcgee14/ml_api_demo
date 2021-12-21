import os
import pytest
import joblib
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Values collected during initial training
initial_scores = {
    "Precision": 0.7284448025785657,
    "Recall": 0.5931758530183727,
    "fbeta": 0.6538878842676311
}

def test_model():
    print(os.listdir('./model'))
    data = joblib.load('./model/artifacts/test.joblib')
    y = data[:, -1]
    X = data[:, :-1]
    model = joblib.load('./model/artifacts/model.joblib')
    
    preds = model.predict(X)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    assert abs(precision - initial_scores['Precision']) <= 0.25
    assert abs(recall - initial_scores['Recall']) <= 0.25
    assert abs(fbeta - initial_scores['fbeta']) <= 0.25