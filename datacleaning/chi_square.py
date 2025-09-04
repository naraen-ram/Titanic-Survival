import pandas as pd
from scipy.stats import chi2_contingency

def run_chi_square_tests(df):
    # age and fare are not categorical so we dont use chi-square test
    # we must compare survived against: gender, SibSp and Parch[after binning], PClass, Embarked, Family

    print("\n---Executing Chi Square Tests for Categorical Variables---\n")

    # Create FamilySize and FamilyCategory(after testing SibSp, Parch: they are weak on their own but more powerful together.)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilyCategory'] = pd.cut(
        df['FamilySize'],
        bins=[0, 1, 4, df['FamilySize'].max()],
        labels=['Alone', 'Small', 'Large']
    )

    tests = {
        "Sex": "Sex",
        "Pclass": "Pclass",
        "Embarked": "Embarked",
        "FamilyCategory": "FamilyCategory"
    }

    for name, col in tests.items():
        contingency_table = pd.crosstab(df[col], df['Survived'])
        print(f"\n--- Survived vs {name} ---")
        print(contingency_table)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print("Chi-Square Value:", chi2)
        print("p-value:", p)
        print("Degrees of Freedom:", dof)
        print("Expected Frequencies:")
        print(expected)

    # best predictors: sex, pclass, family
    # weaker predictors: sibsp, parch, embarked
