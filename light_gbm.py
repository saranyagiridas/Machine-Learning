import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create a small dataset
data = {
    "Age": [22, 25, 28, 35, 40, 45, 30, 50, 27, 38, 42, 24],
    
    "Income": [
        20000, 25000, 30000, 45000,
        50000, 60000, 35000, 70000,
        28000, 48000, 55000, 22000
    ],
    
    "Previous_Purchase": [
        0, 0, 1, 1,
        1, 1, 0, 1,
        0, 1, 1, 0
    ],
    
    "Buy": [
        0, 0, 0, 1,
        1, 1, 0, 1,
        0, 1, 1, 0
    ]
}


# Convert data into DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Separate input and output
X = df[["Age", "Income", "Previous_Purchase"]]
y = df["Buy"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# Create LightGBM model
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbosity=-1
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