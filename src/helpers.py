import joblib
from numpy import concatenate


norm = joblib.load("./model/artifacts/MinMaxScaler.joblib")
encoder = joblib.load("./model/artifacts/OneHotEncoder.joblib")


def get_education_num(row):
    ed_nums = {
        ' Preschool': 1,
        ' 1st-4th': 2,
        ' 5th-6th': 3,
        ' 7th-8th': 4,
        ' 9th': 5,
        ' 10th': 6,
        ' 11th': 7,
        ' 12th': 8,
        ' HS-grad': 9,
        ' Some-college': 10,
        ' Assoc-voc': 11,
        ' Assoc-acdm': 12,
        ' Bachelors': 13,
        ' Masters': 14,
        ' Prof-school': 15,
        ' Doctorate': 16
    }
    return ed_nums.get(row['education'], 0)

def preprocess(feat_df):

    cat_features = [
        "workclass", "education","marital-status","occupation",
        "relationship","race", "sex"
    ]
    X_categorical = feat_df[cat_features]
    X_continuous = feat_df.drop(*[cat_features], axis=1)
    
    X_categorical = encoder.transform(X_categorical)
    X_continuous = norm.transform(X_continuous)
    X = concatenate([X_continuous, X_categorical], axis=1)
    return X

