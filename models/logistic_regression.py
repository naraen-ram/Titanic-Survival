import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/titanic_processed.csv")
#encoding:
df_encoded = pd.get_dummies(df[['Sex', 'Pclass', 'Embarked']], drop_first=True) #encoding each categorical variable
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
X = pd.concat([df[['Fare_log', 'Age', 'FamilySize']], df_encoded], axis=1) # combining the continuous features categorical
y = df['Survived'] # defining the target variable

#splitting train and test datas
# splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

model = LogisticRegression(max_iter=1000)  # increase iterations to ensure convergence
model.fit(X_train, y_train)


y_pred = model.predict(X_test) #predicting on test set

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
