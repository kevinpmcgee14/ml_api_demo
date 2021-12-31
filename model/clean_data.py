import pandas as pd 

def main(df):

    for col in df.columns:
        col1 = col.strip()
        df = df.rename({col: col1}, axis=1)

    df = df[df['capital-gain'] <= 20000]
    df = df[df['capital-loss'] <= 20000]
    df = df[df['fnlgt'] <= 0.75e6]

    df = df.drop(['native-country'], axis=1, errors='ignore')
    df = df[
        (df['workclass'] != 'without-pay') &
        (df['workclass'] != 'never-worked') &
        (df['occupation'] != 'Armed-Forces') &
        (df['marital-status'] != 'Married-AF-spouse')
    ]
    
    return df
    

if __name__ == '__main__':
    df = pd.read_csv('data/census.csv')
    df = main(df)
    df.to_csv('data/census_cleaned.csv', index=None, header=df.columns)