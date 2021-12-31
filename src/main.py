import os


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
import joblib
import json
from pandas import DataFrame
from fastapi import FastAPI
from src.helpers import *
from src.models import Features, Predictions

model = joblib.load("./model/artifacts/model.joblib")
all_cols = [
    'age', 'workclass', 'fnlgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week'
]

app = FastAPI()


@app.get("/")
async def root():
    """Health check to confirm API is reachable.

    Returns:
        message (Dict): Message saying "Hello World"
    """
    return {"message": "Hello World"}



@app.post("/predict/", response_model=Predictions)
async def predict(features: Features):
    """Make a prediction (or multiple) on the salary class for the given features.

    Args:
        features (List[FeatureSet]): An array of FeatureSets, containing all needed columns of data.

    Returns:
        Predictions (List[str]): An array of predictions, where the order matches the order of the features ingested.
    """
    body = json.loads(features.json())
    feature_df = DataFrame(body['features'])
    feature_df.rename(
        {
            'marital_status': 'marital-status', 
            'capital_gain': 'capital-gain', 
            'capital_loss': 'capital-loss', 
            'hours_per_week': 'hours-per-week'
        },
        axis=1, 
        inplace=True
    )
    
    feature_df['education-num'] = feature_df.apply(lambda row: get_education_num(row), axis=1)
    feature_df = feature_df[all_cols]
    X = preprocess(feature_df)
    preds = model.predict(X).tolist()
    response_preds = ['<=50' if pred==0 else '>50' for pred in preds]
    return {"predictions": response_preds}