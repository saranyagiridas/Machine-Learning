import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Create a small dataset
data = {
    "Age": [20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 48, 50],
    "Income": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    "Previous_Purchase": [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    "Buy": [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Separate input and output
X = df[["Age", "Income", "Previous_Purchase"]]
y = df["Buy"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# Create XGBoost model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)

print("\nAccuracy:", accuracy)


# Predict for a new customer
new_customer = [[33, 48, 1]]

prediction = model.predict(new_customer)

print("\nNew Customer Prediction:")

if prediction[0] == 1:
    print("Customer is likely to BUY")
else:
    print("Customer is unlikely to BUY")