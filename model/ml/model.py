import os
import joblib
from ml.data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex"
    ]

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, model='RandomForestClassifier'):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    if model == 'RandomForestClassifier':
        clf = RandomForestClassifier()
    elif model == 'LogisticRegression':
        clf = LogisticRegression(max_iter=500)
    else:
        raise "model not designed to work with repo. Please use LogisticRegression or RandomForestClassifier"
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)



def compute_slices(data, model, feature, norm, ohencoder, lb):

    for ftr in data[feature].unique():
        df = data[data[feature] == ftr]
        if len(df) == 0:
            print(f'{feature} {ftr} had no data.')
            continue
        X, y, norm, ohencoder, lb = process_data(
            df, categorical_features=cat_features, label="salary", training=False, norm=norm, ohencoder=ohencoder, lb=lb
        )

        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        print(f"{feature.upper()}: {ftr.upper()}")
        print("="*20)
        print(f'Number of samples: {len(df)}')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"fbeta: {fbeta:.4f}\n")