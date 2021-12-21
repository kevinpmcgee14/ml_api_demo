# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from halo import Halo

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model

def main():
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
    print('Training model...')
    model = train_model(X_train, y_train)
    
    print('Saving artifacts...')
    joblib.dump(model, './artifacts/model.joblib') 
    joblib.dump(norm, './artifacts/MinMaxScaler.joblib')
    joblib.dump(encoder, './artifacts/OneHotEncoder.joblib')
    joblib.dump(lb, './artifacts/LabelBinarizer.joblib')


if __name__ == '__main__':
    main()