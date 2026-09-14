import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create a small dataset
data = {
    "Age": [22, 25, 28, 35, 40, 45, 30, 50, 27, 38, 42, 24],
    "Gender": [
        "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female",
        "Female", "Male", "Female", "Male"
    ],
    "City": [
        "Kochi", "Thrissur", "Kochi", "Kozhikode",
        "Thrissur", "Kochi", "Kozhikode", "Thrissur",
        "Kochi", "Kozhikode", "Thrissur", "Kochi"
    ],
    "Income": [
        20000, 30000, 25000, 45000,
        50000, 60000, 35000, 70000,
        28000, 48000, 55000, 22000
    ],
    "Buy": [
        "No", "No", "No", "Yes",
        "Yes", "Yes", "Yes", "Yes",
        "No", "Yes", "Yes", "No"
    ]
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Separate input and output
X = df.drop("Buy", axis=1)
y = df["Buy"]

# Categorical columns
categorical_features = ["Gender", "City"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create CatBoost model
model = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.1,
    verbose=False
)

# Train the model
model.fit(
    X_train,
    y_train,
    cat_features=categorical_features
)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred.flatten())

print("\nAccuracy:", accuracy)