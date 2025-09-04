import pandas as pd
from datacleaning.missing_values import handle_missing_values
from datacleaning.outliers import detect_outliers
from datacleaning.transforming import transform_features
# chi-square is optional
# from datacleaning.chi_square import run_chi_square_tests

def preprocess(path="data/Titanic-Dataset.csv", save=False):
    df = pd.read_csv(path)

    df = handle_missing_values(df)
    df = detect_outliers(df)
    df = transform_features(df)

    # run_chi_square_tests(df)   # if you want exploratory results

    if save:
        df.to_csv("data/titanic_processed.csv", index=False)

    return df

if __name__ == "__main__":
    df = preprocess(save=True)
    print("Preprocessing complete.")

