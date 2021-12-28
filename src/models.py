from typing import List, Dict
from pydantic import BaseModel, Field

class FeatureSet(BaseModel):
    age: int
    workclass: str
    fnlgt: float
    education: str
    marital_status: str  = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class Features(BaseModel):
    features: List[FeatureSet]
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class Predictions(BaseModel):
    predictions: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    "<=50"
                ]
            }
        }
    