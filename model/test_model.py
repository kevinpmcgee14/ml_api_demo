import os
import pytest
import joblib
import numpy as np
import pandas as pd
from model.clean_data import main as clean_data
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Values collected during initial training
initial_scores = {
    "Precision": 0.7284448025785657,
    "Recall": 0.5931758530183727,
    "fbeta": 0.6538878842676311
}

def test_clean_data():
    df = pd.read_csv('./data/census.csv', nrows=100)
    df_cleaned = clean_data(df)
    
    removed_count = len(df_cleaned[
        (df_cleaned['workclass'] == 'without-pay') |
        (df_cleaned['workclass'] == 'never-worked') |
        (df_cleaned['occupation'] == 'Armed-Forces') |
        (df_cleaned['marital-status'] == 'Married-AF-spouse')
    ])
    
    assert (df_cleaned['capital-gain'] <= 20000).all()
    assert (df_cleaned['capital-loss'] <= 20000).all()
    assert (df_cleaned['fnlgt'] <= 0.75e6).all()
    assert 'native-country' not in df_cleaned.columns
    assert removed_count == 0
    

def test_predictions():
    data = joblib.load('./model/artifacts/test.joblib')
    y = data[:, -1]
    X = data[:, :-1]
    model = joblib.load('./model/artifacts/model.joblib')
    
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)

def test_metrics():
    data = joblib.load('./model/artifacts/test.joblib')
    y = data[:, -1]
    X = data[:, :-1]
    model = joblib.load('./model/artifacts/model.joblib')
    
    preds = model.predict(X)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    assert precision - initial_scores['Precision'] >= -0.25
    assert recall - initial_scores['Recall'] >= -0.25
    assert fbeta - initial_scores['fbeta'] >= -0.25

