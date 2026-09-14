import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Create house dataset
data = {
    "Area": [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200],
    "Price": [20, 25, 30, 35, 40, 45, 50, 55]
}


# Convert data into DataFrame
df = pd.DataFrame(data)

print("House Dataset:")
print(df)


# Input feature
X = df[["Area"]]

# Target variable
y = df["Price"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# Create Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)


# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)


print("\nMean Squared Error:", mse)
print("R2 Score:", r2)


# Predict price for a new house
new_area = [[1500]]

predicted_price = model.predict(new_area)

print("\nPredicted price for 1500 sq.ft house:")
print(predicted_price[0])


# Plot the data and regression line
plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (Lakhs)")

plt.title("House Price Prediction using Linear Regression")

plt.show()