import pandas as pd
from scipy import stats
import numpy as np
from scipy.stats import chi2_contingency

df = pd.read_csv("Titanic-Dataset.csv") #reads data from csv file and stores it as a dataframe df

missing_count = df.isnull().sum().sort_values(ascending = False)
missing_pct = (df.isnull().mean().round(5).sort_values(ascending=False))*100
missing_df = pd.DataFrame({
    "Missing Values": missing_count,
    "Missing Percent": missing_pct
})

print(missing_df)

#we have almost 77% missing data in cabin column, so we drop it
df = df.drop(columns = ["Cabin"])

#checking is age is skewed
print("Skew of Age: ",df["Age"].skew().round(5)) #0.38911
#age is approximately normal, we check if age is correlated to any other fields
#checking with Pclass and Sex

df["Age_Missing"] = df["Age"].isnull().astype(int)
print(df.groupby("Pclass")["Age_Missing"].mean()*100)
print(df.groupby("Sex")["Age_Missing"].mean()*100)
print(df.groupby("Embarked")["Age_Missing"].mean()*100)
#we found that those embarking in Queensland has more missing age.
#so we have take seperate medians of Pclass+Sex+Embarked : 1+M+C, 1+M+Q, etc... and fill values based on this

# Compute median Age for each (Pclass, Sex, Embarked) group
age_medians = df.groupby(["Pclass", "Sex", "Embarked"])["Age"].median()
print(age_medians)

df["Age"] = df.groupby(["Pclass", "Sex", "Embarked"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
#only 2 missing embarked values, so we drop them
df = df.dropna(subset=["Embarked"])

#now, missing values have been replaced we can check for outliers
#wkt age is basically normal:
age_z = stats.zscore(df["Age"])
age_outliers_flag = np.abs(age_z) >3
age_outliers = df[age_outliers_flag]
print(f"Number of Age outliers (|z| > 3): {age_outliers.shape[0]}")
print(age_outliers[["PassengerId", "Age", "Pclass", "Sex"]])

#fare, sibsp and parch are also numerical categories : use IQR method

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

fare_outliers.head()

#with age, having varying numbers is possible, so we can leave that field.
#we use log transformations for fare - compresses the very large values
#we use binning for SibSp and ParCh - as they are discrete counts: 0, 1-2 and 3+

df['Fare_log'] = np.log1p(df['Fare']) #log(1+fare)

df['SibSp_bin'] = pd.cut(
    df['SibSp'],
    bins=[-1, 0, 2, df['SibSp'].max()],
    labels=['0', '1-2', '3+']
)


df['Parch_bin'] = pd.cut(
    df['Parch'],
    bins=[-1, 0, 2, df['Parch'].max()],
    labels=['0', '1-2', '3+']
)

print(df[['Fare', 'Fare_log', 'SibSp', 'SibSp_bin', 'Parch', 'Parch_bin']].head(10))

#now all outliers have been dealt with
#use chi-square test of independence to check which fields have stronger correspondence to survived...

# age and fare are not categorical so we dont use chi-square test
# we must compare survived against: gender, SibSp and Parch[after binning], PClass, Embarked

# 1. Survived vs Sex
contingency_table = pd.crosstab(df['Sex'], df['Survived'])
print("\n--- Survived vs Sex ---")
print(contingency_table)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Value:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

# 2. Survived vs Pclass
contingency_table = pd.crosstab(df['Pclass'], df['Survived'])
print("\n--- Survived vs Pclass ---")
print(contingency_table)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Value:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

# 3. Survived vs Embarked
contingency_table = pd.crosstab(df['Embarked'], df['Survived'])
print("\n--- Survived vs Embarked ---")
print(contingency_table)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Value:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

# 4. Survived vs SibSp_bin
contingency_table = pd.crosstab(df['SibSp_bin'], df['Survived'])
print("\n--- Survived vs SibSp_bin ---")
print(contingency_table)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Value:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

# 5. Survived vs Parch_bin
contingency_table = pd.crosstab(df['Parch_bin'], df['Survived'])
print("\n--- Survived vs Parch_bin ---")
print(contingency_table)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Value:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

# best predictors: sex, pclass
# weaker predictors: sibsp, parch, embarked
