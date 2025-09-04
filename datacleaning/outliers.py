import numpy as np
from scipy import stats

def detect_outliers(df):
    # now, missing values have been replaced we can check for outliers
    # wkt age is basically normal:
    print("\n---Finding outliers using Z-Score and IQR---\n")
    age_z = stats.zscore(df["Age"])
    age_outliers_flag = np.abs(age_z) > 3
    age_outliers = df[age_outliers_flag]
    print(f"Number of Age outliers (|z| > 3): {age_outliers.shape[0]}")
    print(age_outliers[["PassengerId", "Age", "Pclass", "Sex"]])

    # fare, sibsp and parch are also numerical categories : use IQR method
    def find_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

        print(f"{column} → Outliers: {outliers.shape[0]}")
        return outliers[["PassengerId", column, "Pclass", "Sex"]]

    fare_outliers = find_outliers_iqr(df, "Fare")
    sibsp_outliers = find_outliers_iqr(df, "SibSp")
    parch_outliers = find_outliers_iqr(df, "Parch")

    print(fare_outliers.head())
    return df
