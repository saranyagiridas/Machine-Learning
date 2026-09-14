import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# Create customer data
data = {
    "Annual_Income": [
        15, 16, 17, 18,
        40, 42, 45, 48,
        70, 72, 75, 78
    ],

    "Spending_Score": [
        20, 22, 25, 23,
        50, 52, 48, 55,
        80, 85, 82, 88
    ]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

print("Customer Dataset:")
print(df)


# Select features
X = df[["Annual_Income", "Spending_Score"]]


# Create K-Means model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# Train the model
model.fit(X)


# Get cluster labels
df["Cluster"] = model.labels_


print("\nCustomers with Cluster:")
print(df)


# Display cluster centers
print("\nCluster Centers:")
print(model.cluster_centers_)


# Plot the clusters
plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"]
)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Customer Clustering")

plt.show()