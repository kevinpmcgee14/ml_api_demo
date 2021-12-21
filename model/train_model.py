# Script to train machine learning model.
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def main(model_name):
    # Add code to load in the data.
    data = pd.read_csv('../data/census_cleaned.csv')

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

    # Train and save a model.
    print(f'Training model {model_name} Classifier...')
    model = train_model(X_train, y_train, model=model_name)

    return_metrics(model, X_train, y_train, testing=False)
    return_metrics(model, X_test, y_test, testing=True)
    
    print('Saving artifacts...')
    train_processed = np.concatenate([X_train, np.expand_dims(y_train, axis=1)], axis=1)
    test_processed = np.concatenate([X_test, np.expand_dims(y_test, axis=1)], axis=1)

    joblib.dump(model, './artifacts/model.joblib') 
    joblib.dump(norm, './artifacts/MinMaxScaler.joblib')
    joblib.dump(encoder, './artifacts/OneHotEncoder.joblib')
    joblib.dump(lb, './artifacts/LabelBinarizer.joblib')
    joblib.dump(train_processed, './artifacts/train.joblib')
    joblib.dump(test_processed, './artifacts/test.joblib')


def return_metrics(model, X, y, testing=True):
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    if testing:
        print('TRAINING RESULTS')
    else:
        print('TESTING RESULTS')
    print('='*20)
    print(f"""
        Precision: {precision}\n
        Recall: {recall}\n
        fbeta: {fbeta}\n
        """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="the name of the model to train. Defaults to LogisticRegression",
        default='LogisticRegression')
    args = parser.parse_args()

    main(args.model)