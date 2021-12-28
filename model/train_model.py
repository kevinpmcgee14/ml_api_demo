# Script to train machine learning model.
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slices

def main(model_name, feature_sliced_metrics):
    # Add code to load in the data.
    fdir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv('./data/census_cleaned.csv')

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    #dropped native-country from cleaned dataframe
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex"
    ]
    print('Transforming data...')
    X_train, y_train, norm, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, norm, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, norm=norm, ohencoder=encoder, lb=lb
    )

    train_processed = np.concatenate([X_train, np.expand_dims(y_train, axis=1)], axis=1)
    test_processed = np.concatenate([X_test, np.expand_dims(y_test, axis=1)], axis=1)

    # Train and save a model.
    print(f'Training model {model_name} Classifier...')
    model = train_model(X_train, y_train, model=model_name)

    return_metrics(model, X_train, y_train, testing=False)
    return_metrics(model, X_test, y_test, testing=True)

    print(f'getting metric slices for {feature_sliced_metrics}...')
    compute_slices(test, model, feature_sliced_metrics, norm, encoder, lb)
    
    print('Saving artifacts...')

    joblib.dump(model, os.path.join(fdir, 'artifacts/model.joblib'))
    joblib.dump(norm, os.path.join(fdir, 'artifacts/MinMaxScaler.joblib'))
    joblib.dump(encoder, os.path.join(fdir, 'artifacts/OneHotEncoder.joblib'))
    joblib.dump(lb, os.path.join(fdir, 'artifacts/LabelBinarizer.joblib'))
    joblib.dump(train_processed, os.path.join(fdir, 'artifacts/train.joblib'))
    joblib.dump(test_processed, os.path.join(fdir, 'artifacts/test.joblib'))


def return_metrics(model, X, y, testing=True):
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    if testing:
        print('TESTING RESULTS')
    else:
        print('TRAINING RESULTS')
    print('='*20)
    print(f"""Precision: {precision}\nRecall: {recall}\nfbeta: {fbeta}\n""")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="the name of the model to train. Defaults to LogisticRegression",
        default='LogisticRegression')

    parser.add_argument(
        "--feature_sliced_metrics",
        help="the name of the feature to generate feature slices on. Default is education",
        default='education')
    args = parser.parse_args()

    main(args.model, args.feature_sliced_metrics)