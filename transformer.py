import numpy as np

# Make the results repeatable
np.random.seed(42)

# Four words in a sentence
words = ["I", "love", "this", "movie"]

# Create a small embedding for each word
X = np.random.rand(4, 8)

# Weight matrices for Query, Key and Value
W_Q = np.random.rand(8, 8) * 0.1
W_K = np.random.rand(8, 8) * 0.1
W_V = np.random.rand(8, 8) * 0.1


# Create Query, Key and Value
Q = X @ W_Q
K = X @ W_K
V = X @ W_V


# Calculate attention scores
d_k = 8
scores = (Q @ K.T) / np.sqrt(d_k)


# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Convert scores into attention probabilities
attention_weights = softmax(scores)


# Calculate attention output
attention_output = attention_weights @ V


# Feed-forward layer
W1 = np.random.rand(8, 16) * 0.1
W2 = np.random.rand(16, 8) * 0.1

hidden = np.maximum(0, attention_output @ W1)
feed_forward_output = hidden @ W2


# Add residual connection
output = attention_output + feed_forward_output


# Display results
print("Words:", words)

print("\nInput Shape:")
print(X.shape)

print("\nAttention Weights:")
print(np.round(attention_weights, 3))

print("\nTransformer Output:")
print(np.round(output, 3))

print("\nOutput Shape:")
print(output.shape)