import pandas as pd

def handle_missing_values(df):
    # check missing values
    print("\n---Find missing values and replacing them---\n")
    missing_count = df.isnull().sum().sort_values(ascending = False)
    missing_pct = (df.isnull().mean().round(5).sort_values(ascending=False))*100
    missing_df = pd.DataFrame({
        "Missing Values": missing_count,
        "Missing Percent": missing_pct
    })
    print(missing_df)

    # we have almost 77% missing data in cabin column, so we drop it
    df = df.drop(columns = ["Cabin"])

    # checking is age is skewed
    print("Skew of Age: ",df["Age"].skew().round(5)) #0.38911
    # age is approximately normal, we check if age is correlated to any other fields
    # checking with Pclass and Sex

    df["Age_Missing"] = df["Age"].isnull().astype(int)
    print(df.groupby("Pclass")["Age_Missing"].mean()*100)
    print(df.groupby("Sex")["Age_Missing"].mean()*100)
    print(df.groupby("Embarked")["Age_Missing"].mean()*100)
    # we found that those embarking in Queensland has more missing age.
    # so we have take seperate medians of Pclass+Sex+Embarked : 1+M+C, 1+M+Q, etc... and fill values based on this

    # Compute median Age for each (Pclass, Sex, Embarked) group
    age_medians = df.groupby(["Pclass", "Sex", "Embarked"])["Age"].median()
    print(age_medians)

    df["Age"] = df.groupby(["Pclass", "Sex", "Embarked"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    # only 2 missing embarked values, so we drop them
    df = df.dropna(subset=["Embarked"])

    return df
